import hydra
import numpy as np 
import json
import logging 
# import matplotlib.pyplot as plt # Removed
import os
import openai
import re
import subprocess
from pathlib import Path
import shutil
import time 

from utils.misc import * 
from utils.file_utils import find_files_with_substring, load_tensorboard_logs
from utils.create_task import create_task
from utils.extract_task_code import *
from reward_utils import extract_scalar_parameters, update_reward_function_with_parameters, convert_reward_parameters_to_self_references
from reward_tuner import train_reward_model
from test_policy import capture_rollout, find_latest_checkpoint

EUREKA_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"

PATIENT = True # Block until training is finished when true

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    openai.api_key = os.getenv("OPENAI_API_KEY")

    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model

    print(f"\n{task=}\n", f"{task_description=}\n", f"{suffix=}\n", f"{model=}\n")
    # exit()

    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    env_name = cfg.env.env_name.lower()
    env_parent = 'isaac' if f'{env_name}.py' in os.listdir(f'{EUREKA_ROOT_DIR}/envs/isaac') else 'dexterity'
    task_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py'
    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_code_string  = file_to_string(task_file)
    task_obs_code_string  = file_to_string(task_obs_file)
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

    # Loading all text prompts
    prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    task_code_string = task_code_string.replace(task, task+suffix)
    # Create Task YAML files
    create_task(ISAAC_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None 
    
    # Eureka generation loop
    for iter in range(cfg.iteration):
        # Get Eureka response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = cfg.sample if "gpt-3.5" in model else 4

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        while True:
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    response_cur = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=cfg.temperature,
                        n=chunk_size
                    )
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

            responses.extend(response_cur["choices"])
            prompt_tokens = response_cur["usage"]["prompt_tokens"]
            total_completion_token += response_cur["usage"]["completion_tokens"]
            total_token += response_cur["usage"]["total_tokens"]

        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0]["message"]["content"] + "\n")

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        
        code_runs = [] 
        rl_runs = []
        # This list will store the paths to the *tuned* .py files for this iteration's successful runs
        processed_tuned_code_paths_for_iteration = []

        for response_id in range(cfg.sample):
            response_cur = responses[response_id]["message"]["content"]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string
            # Store the raw extracted code string from LLM after basic cleaning
            raw_llm_code_string_cleaned = ""
            # Remove unnecessary imports
            lines = code_string.split("\n")
            for i, line_idx in enumerate(lines): # Changed variable name to avoid conflict
                if line_idx.strip().startswith("def "):
                    raw_llm_code_string_cleaned = "\n".join(lines[i:])
                    break
            if not raw_llm_code_string_cleaned: # Fallback if "def " not found, use as is
                raw_llm_code_string_cleaned = code_string

            logging.info(f"Iteration {iter}: Code Run {response_id} - Extracted and cleaned LLM code:\n{raw_llm_code_string_cleaned}")

            # --- UNTUNED RUN (if cfg.comparison is True) ---
            if hasattr(cfg, 'comparison') and cfg.comparison:
                logging.info(f"Iteration {iter}: Code Run {response_id} --- Starting UNTUNED run ---")
                try:
                    # Use a distinct variable for the untuned code string
                    code_string_untuned_processing = str(raw_llm_code_string_cleaned)

                    # 1. Extract scalar parameters from the untuned code
                    untuned_scalar_parameters = extract_scalar_parameters(code_string_untuned_processing)
                    logging.info(f"Iteration {iter}: Code Run {response_id} (Untuned) scalar parameters: {untuned_scalar_parameters}")
                    
                    # 2. Update the untuned reward function code with these original parameters (ensures they are embedded)
                    code_string_untuned_final = update_reward_function_with_parameters(code_string_untuned_processing, untuned_scalar_parameters)

                    # 3. Get function signature from the untuned code string
                    gpt_reward_signature_untuned, _ = get_function_signature(code_string_untuned_final)

                    # 4. Prepare task code string with untuned reward
                    untuned_reward_signature_list = [
                        f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature_untuned}",
                        f"self.extras['gpt_reward'] = self.rew_buf.mean()", # Assuming gpt_reward is the one we are comparing
                        f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
                    ]
                    indent = " " * 8
                    untuned_reward_signature_injection = "\n".join([indent + line for line in untuned_reward_signature_list])
                    
                    task_code_string_iter_untuned = ""
                    if "def compute_reward(self)" in task_code_string:
                        task_code_string_iter_untuned = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + untuned_reward_signature_injection)
                    elif "def compute_reward(self, actions)" in task_code_string:
                        task_code_string_iter_untuned = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + untuned_reward_signature_injection)
                    else:
                        raise NotImplementedError("compute_reward signature not found in task_code_string for untuned run.")

                    # 5. Save the new environment code for the untuned run (writes to common output_file)
                    with open(output_file, 'w') as file:
                        file.writelines(task_code_string_iter_untuned + '\n')
                        file.writelines("from typing import Tuple, Dict" + '\n')
                        file.writelines("import math" + '\n')
                        file.writelines("import torch" + '\n')
                        file.writelines("from torch import Tensor" + '\n')
                        code_to_write_untuned = code_string_untuned_final
                        if "@torch.jit.script" not in code_to_write_untuned: # Add jit if not present
                            code_to_write_untuned = "@torch.jit.script\n" + code_to_write_untuned
                        file.writelines(code_to_write_untuned + '\n')
                    
                    # Save bookkeeping copies for untuned run
                    shutil.copy(output_file, f"env_iter{iter}_response{response_id}_untuned.py")
                    with open(f"env_iter{iter}_response{response_id}_untuned_rewardonly.py", 'w') as file:
                        file.writelines(code_string_untuned_final + '\n') # Save the raw reward code

                    # 6. Execute RL for untuned run
                    untuned_rl_filepath = f"env_iter{iter}_response{response_id}_untuned.txt"
                    set_freest_gpu()
                    
                    untuned_process = None
                    with open(untuned_rl_filepath, 'w') as f_untuned:
                        untuned_process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                                        'hydra/output=subprocess',
                                                        f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                                        f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                                                        f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False',
                                                        f'max_iterations={cfg.max_iterations}'], # Consider a different seed or identifier if needed
                                                                stdout=f_untuned, stderr=f_untuned)
                    
                    logging.info(f"Iteration {iter}: Code Run {response_id} (Untuned) training started. Log: {untuned_rl_filepath}")
                    if PATIENT:
                        block_until_training_finished(untuned_rl_filepath, log_status=True, iter_num=iter, response_id=response_id) # Not passing run_label to keep util unchanged
                    else:
                        block_until_training(untuned_rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
                    
                    if untuned_process:
                        untuned_process.communicate() # Ensure it's finished

                    logging.info(f"Iteration {iter}: Code Run {response_id} (Untuned) training finished. Results in {untuned_rl_filepath}")

                except Exception as e_untuned:
                    logging.error(f"Iteration {iter}: Code Run {response_id} (Untuned) run failed: {e_untuned}")
                logging.info(f"Iteration {iter}: Code Run {response_id} --- Finished UNTUNED run ---")

            # --- TUNED RUN (or the only run if not comparing) ---
            # This block uses 'raw_llm_code_string_cleaned' as its starting point.
            logging.info(f"Iteration {iter}: Code Run {response_id} --- Starting TUNED run (or main run) ---")
            try:
                code_string_for_tuning = str(raw_llm_code_string_cleaned) # Use the cleaned raw code

                # First, extract scalar parameters from the reward function
                scalar_parameters = extract_scalar_parameters(code_string_for_tuning)
                logging.info(f"Iteration {iter}: Code Run {response_id} (Tuned Path) initial scalar parameters: {scalar_parameters}")
                
                code_string_with_self_refs = convert_reward_parameters_to_self_references(code_string_for_tuning)

                # PREFERIZE
                # Use getattr for configurable preference tuning parameters
                preference_data_folder = getattr(cfg, "preference_data_folder", "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/preference_data")
                tuning_epochs = getattr(cfg, "tuning_epochs", 5)
                tuning_lr = getattr(cfg, "tuning_lr", 5e-2)

                tuned_reward_model = train_reward_model(
                    code_str=code_string_with_self_refs,
                    param_defaults=scalar_parameters, # Pass original scalar_parameters dict
                    data_folder=preference_data_folder,
                    epochs=tuning_epochs,
                    lr=tuning_lr
                )
                # Update scalar_parameters dictionary with tuned values
                for key in scalar_parameters.keys(): # Iterate over original keys
                    if hasattr(tuned_reward_model, key):
                        scalar_parameters[key] = getattr(tuned_reward_model, key).item()
                
                # Update the reward function code with the tuned parameters
                # The input to update_reward_function_with_parameters should be the one with self-refs
                code_string_tuned_final = update_reward_function_with_parameters(code_string_with_self_refs, scalar_parameters)
                
                gpt_reward_signature, input_lst = get_function_signature(code_string_tuned_final)
                
                code_runs.append(code_string_tuned_final) # This is the tuned code string
                reward_signature_list = [ # Renamed to avoid conflict
                    f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
                    f"self.extras['gpt_reward'] = self.rew_buf.mean()",
                    f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
                ]
                indent = " " * 8
                reward_signature_injection = "\n".join([indent + line for line in reward_signature_list]) # Renamed
                
                task_code_string_iter_tuned = "" # Renamed
                if "def compute_reward(self)" in task_code_string:
                    task_code_string_iter_tuned = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature_injection)
                elif "def compute_reward(self, actions)" in task_code_string:
                    task_code_string_iter_tuned = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature_injection)
                else:
                    raise NotImplementedError("compute_reward signature not found for tuned run.")

                # Save the new environment code for the tuned run (overwrites output_file)
                with open(output_file, 'w') as file:
                    file.writelines(task_code_string_iter_tuned + '\n')
                    file.writelines("from typing import Tuple, Dict" + '\n')
                    file.writelines("import math" + '\n')
                    file.writelines("import torch" + '\n')
                    file.writelines("from torch import Tensor" + '\n')
                    code_to_write_tuned = code_string_tuned_final # Renamed
                    if "@torch.jit.script" not in code_to_write_tuned:
                        code_to_write_tuned = "@torch.jit.script\n" + code_to_write_tuned
                    file.writelines(code_to_write_tuned + '\n')

                bookkeeping_tuned_env_file = f"env_iter{iter}_response{response_id}.py" # Original naming for tuned
                shutil.copy(output_file, bookkeeping_tuned_env_file)
                processed_tuned_code_paths_for_iteration.append(bookkeeping_tuned_env_file)


                with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file: # Original naming
                    file.writelines(code_string_tuned_final + '\n')
            
                set_freest_gpu()
                
                rl_filepath = f"env_iter{iter}_response{response_id}.txt" # Original naming for tuned logs
                process = None
                with open(rl_filepath, 'w') as f:
                    process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                                'hydra/output=subprocess',
                                                f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                                f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                                                f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False',
                                                f'max_iterations={cfg.max_iterations}'],
                
                                                        stdout=f, stderr=f)
                if PATIENT:
                    block_until_training_finished(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
                else:
                    block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
                rl_runs.append(process)

                # Capture a rollout with this policy if the training is successful (for tuned run)
                checkpoint_path = find_latest_checkpoint(task=task,suffix=suffix)
                if checkpoint_path:
                    capture_rollout(seed=2,checkpoint=checkpoint_path,task=task,suffix=suffix)
            
            except Exception as e:
                logging.info(f"Iteration {iter}: Code Run {response_id} (Tuned Path) failed: {e}")
                # No need to append to processed_tuned_code_paths_for_iteration if this path fails before submission
                continue # Skip to next response_id if tuned path fails

        # Gather RL training results and construct reward reflection (for TUNED runs)
        code_feedbacks = []
        contents = []
        successes = []
        reward_correlations = []
        # This list will store the actual paths of successfully processed *tuned* runs for this iteration
        # It will be populated based on `processed_tuned_code_paths_for_iteration` and successful `rl_runs`
        valid_tuned_code_paths_for_feedback = [] 
        
        exec_success = False 
        # Loop over successfully submitted tuned runs
        # The `response_id_idx` is the index into `code_runs` and `rl_runs`
        for response_id_idx, (code_run_content, rl_run_process) in enumerate(zip(code_runs, rl_runs)):
            rl_run_process.communicate()
            
            # Determine the original response_id for file naming. This is tricky.
            # The `processed_tuned_code_paths_for_iteration` list has paths with original response_id.
            # `rl_runs` corresponds to `processed_tuned_code_paths_for_iteration` if no errors before append.
            # Let's assume `processed_tuned_code_paths_for_iteration[response_id_idx]` is the correct path.
            current_tuned_code_path = processed_tuned_code_paths_for_iteration[response_id_idx]
            # The log file path should correspond to this.
            # Example: env_iter0_response5.py -> env_iter0_response5.txt
            rl_filepath_for_gathering = current_tuned_code_path.replace(".py", ".txt")
            # A more robust way to get the log filename, assuming it matches the .py file's base name
            base_log_name = os.path.splitext(os.path.basename(current_tuned_code_path))[0]
            rl_filepath_for_gathering = f"{base_log_name}.txt"


            valid_tuned_code_paths_for_feedback.append(current_tuned_code_path)

            try:
                with open(rl_filepath_for_gathering, 'r') as f: # Use the derived path
                    stdout_str = f.read() 
            except FileNotFoundError:
                logging.error(f"Iteration {iter}: Log file {rl_filepath_for_gathering} not found for a tuned run that was submitted.")
                content = execution_error_feedback.format(traceback_msg=f"Log file {rl_filepath_for_gathering} not found!")
                content += code_output_tip
                contents.append(content) 
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue
            except Exception as e_read: # Other read errors
                logging.error(f"Iteration {iter}: Error reading log file {rl_filepath_for_gathering}: {e_read}")
                content = execution_error_feedback.format(traceback_msg=f"Cannot read log file {rl_filepath_for_gathering} due to {e_read}!")
                content += code_output_tip
                contents.append(content)
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ''
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == '':
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                lines = stdout_str.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Tensorboard Directory:'):
                        break 
                tensorboard_logdir = line.split(':')[-1].strip() 
                tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                epoch_freq = max(int(max_iterations // 10), 1)
                
                content += policy_feedback.format(epoch_freq=epoch_freq)
                
                # Compute Correlation between Human-Engineered and GPT Rewards
                if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                    gt_reward = np.array(tensorboard_logs["gt_reward"])
                    gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_correlations.append(reward_correlation)

                # Add reward components log to the feedback
                for metric in tensorboard_logs:
                    if "/" not in metric:
                        metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                        metric_cur_max = max(tensorboard_logs[metric])
                        metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                        if "consecutive_successes" == metric:
                            successes.append(metric_cur_max)
                        metric_cur_min = min(tensorboard_logs[metric])
                        if metric != "gt_reward" and metric != "gpt_reward":
                            if metric != "consecutive_successes":
                                metric_name = metric 
                            else:
                                metric_name = "task_score"
                            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                        else:
                            # Provide ground-truth score when success rate not applicable
                            if "consecutive_successes" not in tensorboard_logs:
                                content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                code_feedbacks.append(code_feedback)
                content += code_feedback  
            else:
                # Otherwise, provide execution traceback error feedback
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content) 
        
        # Select the best code sample based on the success rate (from tuned runs)
        best_sample_idx = np.argmax(np.array(successes)) if successes else -1 # Ensure successes is not empty
        
        if best_sample_idx != -1: # If there were any successful tuned runs
            best_content = contents[best_sample_idx]
            max_success = successes[best_sample_idx]
            max_success_reward_correlation = reward_correlations[best_sample_idx]
            # Ensure valid_tuned_code_paths_for_feedback corresponds to successes
            current_best_tuned_code_path = valid_tuned_code_paths_for_feedback[best_sample_idx]

            # Update the best Eureka Output (based on tuned runs)
            if max_success > max_success_overall:
                max_success_overall = max_success
                max_success_reward_correlation_overall = max_success_reward_correlation
                max_reward_code_path = current_best_tuned_code_path # Path to the best *tuned* code

            execute_rate = np.sum(np.array(successes) >= 0.) / cfg.sample # Recalculate execute_rate based on actual successes
            max_successes.append(max_success)
            max_successes_reward_correlation.append(max_success_reward_correlation)
            best_code_paths.append(current_best_tuned_code_path)

            logging.info(f"Iteration {iter}: Max Success (Tuned): {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation (Tuned): {max_success_reward_correlation}")
            logging.info(f"Iteration {iter}: Best Generation ID (within this iter, for tuned): {best_sample_idx}")
            # The response for LLM feedback should be the one that led to this best *tuned* result.
            # This requires mapping best_sample_idx back to original response_id if responses were skipped.
            # For simplicity, responses[best_sample_idx] assumes direct mapping if no skips,
            # or that `successes` list aligns with `responses` if failures append DUMMY_FAILURE.
            # The original code uses responses[best_sample_idx], so we keep that.
            # To be more robust, we'd need to track original response_ids alongside successes.
            # For now, assuming `responses` and `successes` align for the best_sample_idx.
            original_response_index_for_best = -1
            if valid_tuned_code_paths_for_feedback:
                 # Try to extract original response_id from the filename if possible, e.g. env_iterX_responseY.py
                try:
                    match = re.search(r"_response(\d+)", os.path.basename(current_best_tuned_code_path))
                    if match:
                        original_response_index_for_best = int(match.group(1))
                except:
                    pass # Keep -1 if parsing fails

            if original_response_index_for_best != -1 and original_response_index_for_best < len(responses):
                 logging.info(f"Iteration {iter}: GPT Output Content (for best tuned run, original_id={original_response_index_for_best}):\n" +  responses[original_response_index_for_best]["message"]["content"] + "\n")
                 # Update messages for LLM
                 if len(messages) == 2:
                     messages += [{"role": "assistant", "content": responses[original_response_index_for_best]["message"]["content"]}]
                     messages += [{"role": "user", "content": best_content}]
                 else:
                     assert len(messages) == 4
                     messages[-2] = {"role": "assistant", "content": responses[original_response_index_for_best]["message"]["content"]}
                     messages[-1] = {"role": "user", "content": best_content}
            else: # Fallback if original_id cannot be determined or is out of bounds
                 logging.warning(f"Iteration {iter}: Could not reliably map best_sample_idx to original response for LLM feedback. Using best_sample_idx={best_sample_idx} as index into responses.")
                 if best_sample_idx < len(responses): # Check bounds for safety
                    logging.info(f"Iteration {iter}: GPT Output Content (for best tuned run, fallback idx={best_sample_idx}):\n" +  responses[best_sample_idx]["message"]["content"] + "\n")
                    if len(messages) == 2:
                        messages += [{"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}]
                        messages += [{"role": "user", "content": best_content}]
                    else:
                        assert len(messages) == 4
                        messages[-2] = {"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}
                        messages[-1] = {"role": "user", "content": best_content}
                 else:
                    logging.error(f"Iteration {iter}: best_sample_idx {best_sample_idx} is out of bounds for responses list (len {len(responses)}). Cannot update LLM messages.")


            logging.info(f"Iteration {iter}: User Content (feedback for best tuned run):\n" + best_content + "\n")
            
        else: # No successful tuned runs in this iteration
            execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info(f"Iteration {iter}: No successful tuned runs to select from. LLM messages will not be updated with new feedback for this iteration.")
            # If all code generation failed (exec_success is False), the original code continues.
            # This 'else' handles when there are no successes to pick a best from.

        # Plot the success rate (based on tuned runs) - Plotting section removed
        # fig, axs = plt.subplots(2, figsize=(6, 6))
        # fig.suptitle(f'{cfg.env.task}')

        # x_axis = np.arange(len(max_successes))

        # axs[0].plot(x_axis, np.array(max_successes))
        # axs[0].set_title("Max Success")
        # axs[0].set_xlabel("Iteration")

        # axs[1].plot(x_axis, np.array(execute_rates))
        # axs[1].set_title("Execute Rate")
        # axs[1].set_xlabel("Iteration")

        # fig.tight_layout(pad=3.0)
        # plt.savefig('summary.png') # Removed
        np.savez('summary.npz', max_successes=max_successes, execute_rates=execute_rates, best_code_paths=best_code_paths, max_successes_reward_correlation=max_successes_reward_correlation)

        # The messages list has already been updated correctly by the logic
        # within the `if best_sample_idx != -1:` block (or left unchanged if no success).
    
    # Evaluate the best reward code many times
    if max_reward_code_path is None: 
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    logging.info(f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}")
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    shutil.copy(max_reward_code_path, output_file)
    
    eval_runs = []
    for i in range(cfg.num_eval):
        set_freest_gpu()
        
        # Execute the python file with flags
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                        'hydra/output=subprocess',
                                        f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                        f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                                        f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False', f'seed={i}',
                                        ],
                                        stdout=f, stderr=f)


        if PATIENT:
            block_until_training_finished(rl_filepath)
        else:
            block_until_training(rl_filepath)
        eval_runs.append(process)

    reward_code_final_successes = []
    reward_code_correlations_final = []
    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate()
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'r') as f:
            stdout_str = f.read() 
        lines = stdout_str.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('Tensorboard Directory:'):
                break 
        tensorboard_logdir = line.split(':')[-1].strip() 
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_success = max(tensorboard_logs['consecutive_successes'])
        reward_code_final_successes.append(max_success)

        if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
            gt_reward = np.array(tensorboard_logs["gt_reward"])
            gpt_reward = np.array(tensorboard_logs["gpt_reward"])
            reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
            reward_code_correlations_final.append(reward_correlation)

    logging.info(f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}")
    logging.info(f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}")
    np.savez('final_eval.npz', reward_code_final_successes=reward_code_final_successes, reward_code_correlations_final=reward_code_correlations_final)


if __name__ == "__main__":
    # Arg patient
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('patient', type=str, help='Patient ID') Patient ID is probably not the correct patient assumption
    main()