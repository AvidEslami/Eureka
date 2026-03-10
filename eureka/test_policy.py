import sys
import os
import signal
import subprocess
import logging
import shutil
from eureka import ISAAC_ROOT_DIR, EUREKA_ROOT_DIR
from utils.misc import *
from pathlib import Path

EUREKA_CONDA_ENV = "eureka"
CONDA_PYTHON = ['conda', 'run', '-n', EUREKA_CONDA_ENV, '--no-capture-output', 'python']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout, force=True)

def find_latest_checkpoint(task, suffix):
    """
    Finds the most recent checkpoint file (.pth) for a given task and suffix
    by recursively searching the EUREKA_ROOT_DIR.
    """
    run_prefix = f"{task}{suffix}"
    search_pattern = f"**/runs/{run_prefix}*/nn/*.pth"
    checkpoint_paths = sorted(Path(EUREKA_ROOT_DIR).rglob(search_pattern))
    
    if not checkpoint_paths:
        logging.warning(f"No checkpoint found for task={task}, suffix={suffix}.")
        return None

    latest_checkpoint = str(checkpoint_paths[-1])
    logging.info(f"Using checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def deploy_rollout(seed=1, task="ShadowHandSpin", suffix="", checkpoint=f"{ISAAC_ROOT_DIR}/checkpoints/EurekaPenSpinning.pth", capture_video=False, headless=False):
    '''
    The goal of this function is to deploy a rollout of the policy on the environment and return the Fitness of the rollout.
        This fitness can be tentatively used to determine preference pairs.
    
    Manual Deploy Command Example:
    python train.py test=True headless=False force_render=True task=ShadowHandSpin checkpoint=checkpoints/EurekaPenSpinning.pth 
    '''
    
    rl_filepath = f"reward_code_eval_deploy_testing.txt"    
    with open(rl_filepath, 'w') as f:
        if task == "ShadowHand" or task == "ShadowHandBottleCap" or task == "ShadowHandDoorOpenInward":
            process = subprocess.Popen([*CONDA_PYTHON, '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                        'hydra/output=subprocess',
                                        f'test=True', f'checkpoint={checkpoint}',
                                        f'task={task}{suffix}',
                                        f'headless={headless}', f'capture_video={capture_video}', 'force_render={headless}', f'seed={seed}', 
                                        f'task.env.printNumSuccesses=True' ,
                                        ],
                                        stdout=f, stderr=f)
        else:
            process = subprocess.Popen([*CONDA_PYTHON, '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                        'hydra/output=subprocess',
                                        f'test=True', f'checkpoint={checkpoint}',
                                        f'task={task}{suffix}',
                                        f'headless={not capture_video}', f'capture_video={capture_video}', 'force_render=False', f'seed={seed}',                                        ],
                                        stdout=f, stderr=f)
        success_score = block_until_finished_testing(rl_filepath, log_status=True)
        while True:
            # Wait endlessely until the process is done
            retcode = process.poll()
            if retcode is not None:
                break
        # Terminate the process after capturing the rollout
        process.kill()
        print(f"Process Completed. Success Score: {success_score}")
        return success_score  # Return the extracted success metric

def capture_rollout(seed=2, task=None, suffix="", checkpoint=f"{ISAAC_ROOT_DIR}/checkpoints/EurekaPenSpinning.pth", capture_video=False, rl_filepath = "reward_code_eval_deploy_testing.txt", log_status=False):
    '''
    The goal of this function is to deploy a rollout of the policy on the environment and save the list of states reached.
    
    Manual Deploy Command Example:
    python train.py test=True headless=False force_render=True task=ShadowHandSpin checkpoint=checkpoints/EurekaPenSpinning.pth 
    '''
    if task in ("ShadowHand", "ShadowHandBottleCap", "ShadowHandDoorOpenInward", "ShadowHandDoorOpenOutward"):
        # rl_filepath = f"reward_code_eval_deploy_testing.txt"    
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen([*CONDA_PYTHON, '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                        'hydra/output=subprocess',
                                        f'test=True', f'checkpoint={checkpoint}',
                                        f'task={task}{suffix}',
                                        f'headless={not capture_video}', f'capture_video={capture_video}', 'force_render=True', f'seed={seed}', 
                                        f'task.env.printNumSuccesses=False'#, f'from_data=False'
                                        ],
                                        stdout=f, stderr=f)
            stop_at = 1.0
            for success_value in monitor_direct_success(rl_filepath, process, log_status=log_status):
                    print(f"Current success: {success_value}")
                    if success_value == stop_at:
                        print("Success achieved!")
                        process.terminate()
                        try: 
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            print("Process did not terminate in time, killing it.")
                            process.kill()
                        stop_at_success = True
                        break
            else:
                success_value = None

            stop_at_success = False
            # time.sleep(0.1)
            success_score = block_until_rollout_captured(rl_filepath, log_status=True, task_name=task, stop_at_success=stop_at_success, seed=seed, success_reached=success_value, capture_video=capture_video)
            print(f"Process Completed. Success Score: {success_score}")
            return success_score  # Return the extracted success metric
    elif task in ("ShadowHandScissors"):
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen([*CONDA_PYTHON, '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                        'hydra/output=subprocess',
                                        f'test=True', f'checkpoint={checkpoint}',
                                        f'task={task}{suffix}',
                                        f'headless={not capture_video}', f'capture_video={capture_video}', 'force_render=False', f'seed={seed}', 
                                        f'task.env.printNumSuccesses=True' ,
                                        ],
                                        stdout=f, stderr=f)
            success_score = block_until_rollout_captured(rl_filepath, log_status=True, task_name=task, seed=seed, capture_video=capture_video)
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                print("Process did not terminate in time, killing it.")
                process.kill()
            print(f"Process Completed. Success Score: {success_score}")
            return success_score
    else:
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen([*CONDA_PYTHON, '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                        'hydra/output=subprocess',
                                        f'test=True', f'checkpoint={checkpoint}',
                                        f'task={task}{suffix}',
                                        f'headless={not capture_video}', f'capture_video={capture_video}', 'force_render=False', f'seed={seed}', 
                                        f'task.env.printSuccessStat=True' ,
                                        ],
                                        stdout=f, stderr=f)
            max_steps = 100
            average_success_score = block_until_rollout_captured(rl_filepath, log_status=True, task_name=task, stop_at_success=False, seed=seed, max_steps=max_steps, capture_video=capture_video)
            # Terminate the process after capturing the rollout
            # print(f"Process Completed. Average Success Score: {average_success_score}")
            process.terminate()
            try: 
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Process did not terminate in time, killing it.")
                process.kill()
            return average_success_score  # Return the extracted success metric
        


def capture_reward_from_rollout(data_list_path, seed=2, task="ShadowHandSpin", suffix="", checkpoint=f"{ISAAC_ROOT_DIR}/checkpoints/EurekaPenSpinning.pth", capture_video=False):
    '''
    The goal of this function is to deploy a rollout of the policy on the environment and save the list of states reached.
    
    Manual Deploy Command Example:
    python train.py test=True headless=False force_render=True task=ShadowHandSpin checkpoint=checkpoints/EurekaPenSpinning.pth 
    '''
    # Open the data_list file and read the data, skipping the first line
    # with open(data_list_path, 'r') as f:
    #     data_list = f.readlines()[1:]
    


    rl_filepath = f"reward_code_eval_deploy_testing.txt"    
    with open(rl_filepath, 'w') as f:
        process = subprocess.Popen([*CONDA_PYTHON, '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                    'hydra/output=subprocess',
                                    f'test=True', f'checkpoint={checkpoint}',
                                    f'task={task}{suffix}',
                                    f'headless={not capture_video}', f'capture_video={capture_video}', 'force_render=False', f'seed={seed}', 
                                    f'from_data=True', f'data_list={data_list_path}',
                                    f'task.env.printNumSuccesses=True'
                                    ],
                                    stdout=f, stderr=f)
        success_score = block_until_rollout_captured(rl_filepath, log_status=True, task_name=task, capture_video=capture_video)
        print(f"Process Completed. Success Score: {success_score}")
    return

def deploy_train(seed=1, task="ShadowHandSpin", suffix="", max_iterations=1000, checkpoint=f"{ISAAC_ROOT_DIR}/checkpoints/EurekaPenSpinning.pth", capture_video=False, rl_filepath="reward_code_eval_deploy_testing.txt",
                 wandb_activate=True, wandb_entity="george-xue-university-of-toronto", wandb_project="PEureka"):
    '''
    The goal of this function is to deploy a rollout of the policy on the environment and return the Fitness of the rollout.
        This fitness can be tentatively used to determine preference pairs.

    Manual Deploy Command Example:
    python train.py test=True headless=False force_render=True task=ShadowHandSpin checkpoint=checkpoints/EurekaPenSpinning.pth
    '''

    with open(rl_filepath, 'w') as f:
        process = subprocess.Popen([*CONDA_PYTHON, '-u', f'{ISAAC_ROOT_DIR}/train.py',
                                    'hydra/output=subprocess',
                                    f'task={task}{suffix}',
                                    f'headless={not capture_video}', f'capture_video={capture_video}', 'force_render=False', f'seed={seed}',
                                    f'max_iterations={max_iterations}',
                                    f'wandb_activate={wandb_activate}', f'wandb_entity={wandb_entity}', f'wandb_project={wandb_project}',
                                    ],
                                    stdout=f, stderr=f)
        success_score = block_until_training_finished(rl_filepath, log_status=True)
        print(f"Process Completed. Success Score: {success_score}")
        return success_score  # Return the extracted success metric

if __name__ == "__main__":
    # # =========================================================================
    # # Train with Eureka reported reward functions (rollout suffix)
    # # =========================================================================
    # deploy_train(
    #     seed=42,
    #     task="ShadowHandDoorOpenOutward",
    #     suffix="rollout",
    #     max_iterations=2000,
    #     rl_filepath="train_door_open_outward_rollout.txt",
    # )

    # deploy_train(
    #     seed=42,
    #     task="ShadowHandDoorOpenInward",
    #     suffix="rollout",
    #     max_iterations=2000,
    #     rl_filepath="train_door_open_inward_rollout.txt",
    # )

    # =========================================================================
    # Collect 100 evenly spaced rollouts (with video) for each task.
    # Videos are copied into {data_folder}/videos/ so they survive policy cleanup.
    # =========================================================================
    rollout_jobs = [
        {
            "task": "ShadowHandDoorOpenOutward",
            "suffix": "rollout",
            "train_log": "train_door_open_outward_rollout.txt",
            "data_folder": Path("/home/gx22/Desktop/isaacgym/python/Eureka/eureka/auto_preference_data_open_outward"),
            "log_dir": Path("/home/gx22/Desktop/isaacgym/python/Eureka/eureka/door_outward_rollout_logs"),
        },
        # {
        #     "task": "ShadowHandDoorOpenInward",
        #     "suffix": "rollout",
        #     "train_log": "train_door_open_inward_rollout.txt",
        #     "data_folder": Path("/home/gx22/Desktop/isaacgym/python/Eureka/eureka/auto_preference_data_open_inward"),
        #     "log_dir": Path("/home/gx22/Desktop/isaacgym/python/Eureka/eureka/door_inward_rollout_logs"),
        # },
    ]

    seed = 42
    num_rollouts = 100
    start_epoch = 10
    end_epoch = 2000
    auto_pref_dir = Path("/home/gx22/Desktop/isaacgym/python/Eureka/eureka/auto_preference_data")

    for job in rollout_jobs:
        task = job["task"]
        suffix = job["suffix"]
        data_folder = job["data_folder"]
        log_dir = job["log_dir"]

        data_folder.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        auto_pref_dir.mkdir(exist_ok=True)
        video_dir = data_folder / "videos"
        video_dir.mkdir(exist_ok=True)

        # Find the nn directory from the training log
        train_log = file_to_string(job["train_log"])
        nn_dir = None
        for line in train_log.split('\n'):
            if 'Network Directory:' in line:
                nn_dir = Path(line.split('Network Directory:')[1].strip())
                break

        if nn_dir is None or not nn_dir.exists():
            print(f"ERROR: Could not find nn directory for {task} from {job['train_log']}")
            continue

        # Compute 100 evenly spaced epochs
        epochs = [round(start_epoch + i * (end_epoch - start_epoch) / (num_rollouts - 1))
                  for i in range(num_rollouts)]

        print(f"\n{'='*60}")
        print(f"Collecting {num_rollouts} rollouts with video for {task}")
        print(f"  Checkpoint dir : {nn_dir}")
        print(f"  Data folder    : {data_folder}")
        print(f"  Video folder   : {video_dir}")
        print(f"  Log dir        : {log_dir}")
        print(f"  Epochs ({len(epochs)}): {epochs[:5]}...{epochs[-5:]}")
        print(f"{'='*60}\n")

        task_class_name = f"{task}{suffix}"
        collected = 0
        skipped = 0
        for idx, epoch in enumerate(epochs):
            # Find checkpoint for this epoch
            matches = list(nn_dir.glob(f"{task_class_name}_successes_{epoch}_*.pth"))
            if not matches and epoch == end_epoch:
                matches = list(nn_dir.glob(f"last_{task_class_name}_ep_{epoch}*.pth"))
            if not matches:
                print(f"[{idx+1}/{num_rollouts}] Epoch {epoch}: checkpoint not found, skipping")
                skipped += 1
                continue

            checkpoint_path = str(matches[0])
            rl_filepath = str(log_dir / f"rollout_epoch{epoch}.txt")

            print(f"\n[{idx+1}/{num_rollouts}] Epoch {epoch}")
            print(f"  Checkpoint: {Path(checkpoint_path).name}")

            try:
                capture_rollout(
                    seed=seed,
                    task=task,
                    suffix="",  # capture_rollout uses base task (has print statements)
                    checkpoint=checkpoint_path,
                    capture_video=True,
                    rl_filepath=rl_filepath,
                    log_status=True,
                )
                collected += 1
                print(f"  Rollout captured ({collected} so far)")
            except Exception as e:
                print(f"  Failed: {e}")
                skipped += 1
                continue

            # Move rollout data from auto_preference_data/ to target data folder,
            # and copy the video into {data_folder}/videos/ so it survives policy cleanup.
            for src_file in sorted(auto_pref_dir.glob(f"*{task}*.txt")):
                # Read first line to check for video path
                with open(src_file, 'r') as f:
                    first_line = f.readline().strip()
                    rest = f.read()

                new_video_path = ""
                if first_line.startswith("/") and os.path.isfile(first_line):
                    # Copy video to persistent location
                    video_name = f"epoch{epoch}_{src_file.stem}.mp4"
                    dest_video = video_dir / video_name
                    shutil.copy2(first_line, dest_video)
                    new_video_path = str(dest_video)
                    print(f"  Video saved: {dest_video.name}")

                # Write data file to target folder with updated video path
                dst_file = data_folder / src_file.name
                with open(dst_file, 'w') as f:
                    if new_video_path:
                        f.write(f"{new_video_path}\n")
                    else:
                        f.write(f"{first_line}\n")
                    f.write(rest)

                # Remove original from auto_preference_data
                src_file.unlink()

        print(f"\n{'='*60}")
        print(f"DONE: {task}")
        print(f"  Collected : {collected}/{num_rollouts}")
        print(f"  Skipped   : {skipped}")
        print(f"  Data in   : {data_folder}")
        print(f"  Videos in : {video_dir}")
        print(f"  Logs in   : {log_dir}")
        print(f"{'='*60}")

    # # =========================================================================
    # # Train ShadowHandDoorOpenOutward with ground truth reward (no GPT suffix)
    # # =========================================================================
    # deploy_train(
    #     seed=42,
    #     task="ShadowHandDoorOpenOutward",
    #     suffix="",
    #     max_iterations=6000,
    #     rl_filepath="train_door_open_outward_gt.txt",
    # )

    # =========================================================================
    # PREVIOUS: Collect 75 evenly spaced rollouts (with video) for
    # ShadowHandDoorOpenInward from iteration 1 (epochs 10–2000).
    # =========================================================================
    # import shutil
    # task = "ShadowHandDoorOpenInward"
    # seed = 42
    # nn_dir = Path(
    #     "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/experiments/"
    #     "ShadowHandDoorOpenInward_2026-02-13_02-33-56/iteration_1/policy/runs/"
    #     "ShadowHandDoorOpenInwardGPT-2026-02-13_02-42-17/nn"
    # )
    # log_dir = Path("/home/gx22/Desktop/isaacgym/python/Eureka/eureka/door_inward_rollout_logs")
    # log_dir.mkdir(exist_ok=True)
    #
    # source_dir = Path("/home/gx22/Desktop/isaacgym/python/Eureka/eureka/auto_preference_data")
    # target_dir = Path("/home/gx22/Desktop/isaacgym/python/Eureka/eureka/auto_preference_data_open_inward")
    # source_dir.mkdir(exist_ok=True)
    # target_dir.mkdir(exist_ok=True)
    #
    # num_rollouts = 75
    # start_epoch = 10
    # end_epoch = 2000
    # epochs = [round(start_epoch + i * (end_epoch - start_epoch) / (num_rollouts - 1))
    #           for i in range(num_rollouts)]
    #
    # print(f"{'='*60}")
    # print(f"Collecting {num_rollouts} rollouts with video for {task}")
    # print(f"Checkpoint dir : {nn_dir}")
    # print(f"Log dir        : {log_dir}")
    # print(f"Source data dir : {source_dir}")
    # print(f"Target data dir : {target_dir}")
    # print(f"Epochs ({len(epochs)}): {epochs}")
    # print(f"{'='*60}\n")
    #
    # collected = 0
    # skipped = 0
    # for idx, epoch in enumerate(epochs):
    #     matches = list(nn_dir.glob(f"ShadowHandDoorOpenInwardGPT_successes_{epoch}_*.pth"))
    #     if not matches and epoch == end_epoch:
    #         matches = list(nn_dir.glob(f"last_ShadowHandDoorOpenInwardGPT_ep_{epoch}.pth"))
    #     if not matches:
    #         print(f"[{idx+1}/{num_rollouts}] Epoch {epoch}: checkpoint not found, skipping")
    #         skipped += 1
    #         continue
    #
    #     checkpoint_path = str(matches[0])
    #     rl_filepath = str(log_dir / f"rollout_epoch{epoch}.txt")
    #
    #     print(f"\n[{idx+1}/{num_rollouts}] Epoch {epoch}")
    #     print(f"  Checkpoint: {Path(checkpoint_path).name}")
    #
    #     try:
    #         capture_rollout(
    #             seed=seed,
    #             task=task,
    #             suffix="",
    #             checkpoint=checkpoint_path,
    #             capture_video=True,
    #             rl_filepath=rl_filepath,
    #             log_status=True,
    #         )
    #         collected += 1
    #         print(f"  Rollout captured ({collected} so far)")
    #     except Exception as e:
    #         print(f"  Failed: {e}")
    #         skipped += 1
    #         continue
    #
    # print(f"\n{'='*60}")
    # print(f"Moving rollout data from {source_dir} -> {target_dir}")
    # moved = 0
    # for src_file in sorted(source_dir.glob(f"*{task}*.txt")):
    #     dst_file = target_dir / src_file.name
    #     if dst_file.exists():
    #         import datetime as _dt
    #         ts = _dt.datetime.now().strftime("%Y%m%d%H%M%S")
    #         dst_file = target_dir / f"{src_file.stem}_{ts}{src_file.suffix}"
    #     shutil.move(str(src_file), str(dst_file))
    #     moved += 1
    #
    # print(f"\n{'='*60}")
    # print(f"DONE")
    # print(f"  Collected : {collected}/{num_rollouts}")
    # print(f"  Skipped   : {skipped}")
    # print(f"  Moved     : {moved} data files -> {target_dir}")
    # print(f"  Logs in   : {log_dir}")
    # print(f"{'='*60}")

    # =========================================================================
    # PREVIOUS: Record videos of the final policy from two experiments.
    # Uncomment to use.
    # =========================================================================
    # jobs = [
    #     {
    #         "task": "ShadowHandDoorOpenInward",
    #         "checkpoint": "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/experiments/"
    #                       "ShadowHandDoorOpenInward_2026-02-13_02-33-56/iteration_5/policy/runs/"
    #                       "ShadowHandDoorOpenInwardGPT-2026-02-13_06-01-02/nn/"
    #                       "ShadowHandDoorOpenInwardGPT_successes_1443_0.00.pth",
    #     },
    #     {
    #         "task": "ShadowHandDoorOpenOutward",
    #         "checkpoint": "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/experiments/"
    #                       "ShadowHandDoorOpenOutward_2026-02-13_02-32-52/iteration_5/policy/runs/"
    #                       "ShadowHandDoorOpenOutwardGPT-2026-02-13_06-19-18/nn/"
    #                       "ShadowHandDoorOpenOutwardGPT_successes_1078_0.86.pth",
    #     },
    # ]
    #
    # for job in jobs:
    #     task = job["task"]
    #     ckpt = job["checkpoint"]
    #     log_file = f"video_capture_{task}.txt"
    #
    #     print(f"\n{'='*60}")
    #     print(f"Recording video for {task}")
    #     print(f"  Checkpoint: {ckpt}")
    #     print(f"{'='*60}")
    #
    #     with open(log_file, 'w') as f:
    #         process = subprocess.Popen(
    #             [*CONDA_PYTHON, '-u', f'{ISAAC_ROOT_DIR}/train.py',
    #              'hydra/output=subprocess',
    #              'test=True',
    #              f'checkpoint={ckpt}',
    #              f'task={task}',
    #              'headless=False',
    #              'capture_video=True',
    #              'force_render=True',
    #              'seed=42',
    #              ],
    #             stdout=f, stderr=f,
    #             cwd=EUREKA_ROOT_DIR,
    #         )
    #         process.wait()
    #
    #     print(f"  Done (exit code {process.returncode}). Log: {log_file}")
    #
    # print(f"\n{'='*60}")
    # print("All done. Check eureka/ for policy-* folders containing videos/")
    # print(f"{'='*60}")

    # =========================================================================
    # PREVIOUS: Collect 75 evenly spaced rollouts (with video) for
    # ShadowHandDoorOpenOutward. Uncomment to use.
    # =========================================================================
    # task = "ShadowHandDoorOpenOutward"
    # seed = 42
    # nn_dir = Path(
    #     "/home/gx22/Desktop/isaacgym/python/Eureka/isaacgymenvs/isaacgymenvs/"
    #     "policy-2026-01-29_12-20-07/runs/"
    #     "ShadowHandDoorOpenOutwardGPT-2026-01-29_12-20-07/nn"
    # )
    # log_dir = Path("/home/gx22/Desktop/isaacgym/python/Eureka/eureka/door_outward_rollout_logs")
    # log_dir.mkdir(exist_ok=True)
    #
    # source_dir = Path("/home/gx22/Desktop/isaacgym/python/Eureka/eureka/auto_preference_data")
    # target_dir = Path("/home/gx22/Desktop/isaacgym/python/Eureka/eureka/auto_preference_data_open_outward")
    # source_dir.mkdir(exist_ok=True)
    # target_dir.mkdir(exist_ok=True)
    #
    # num_rollouts = 75
    # start_epoch = 10
    # end_epoch = 1000
    # epochs = [round(start_epoch + i * (end_epoch - start_epoch) / (num_rollouts - 1))
    #           for i in range(num_rollouts)]
    #
    # print(f"{'='*60}")
    # print(f"Collecting {num_rollouts} rollouts with video for {task}")
    # print(f"Checkpoint dir : {nn_dir}")
    # print(f"Log dir        : {log_dir}")
    # print(f"Source data dir : {source_dir}")
    # print(f"Target data dir : {target_dir}")
    # print(f"Epochs ({len(epochs)}): {epochs}")
    # print(f"{'='*60}\n")
    #
    # collected = 0
    # skipped = 0
    # for idx, epoch in enumerate(epochs):
    #     matches = list(nn_dir.glob(f"ShadowHandDoorOpenOutwardGPT_successes_{epoch}_*.pth"))
    #     if not matches and epoch == end_epoch:
    #         matches = list(nn_dir.glob(f"last_ShadowHandDoorOpenOutwardGPT_ep_{epoch}.pth"))
    #     if not matches:
    #         print(f"[{idx+1}/{num_rollouts}] Epoch {epoch}: checkpoint not found, skipping")
    #         skipped += 1
    #         continue
    #
    #     checkpoint_path = str(matches[0])
    #     rl_filepath = str(log_dir / f"rollout_epoch{epoch}.txt")
    #
    #     print(f"\n[{idx+1}/{num_rollouts}] Epoch {epoch}")
    #     print(f"  Checkpoint: {Path(checkpoint_path).name}")
    #
    #     try:
    #         capture_rollout(
    #             seed=seed,
    #             task=task,
    #             suffix="",
    #             checkpoint=checkpoint_path,
    #             capture_video=True,
    #             rl_filepath=rl_filepath,
    #             log_status=True,
    #         )
    #         collected += 1
    #         print(f"  Rollout captured ({collected} so far)")
    #     except Exception as e:
    #         print(f"  Failed: {e}")
    #         skipped += 1
    #         continue
    #
    # print(f"\n{'='*60}")
    # print(f"Moving rollout data from {source_dir} -> {target_dir}")
    # moved = 0
    # for src_file in sorted(source_dir.glob(f"*{task}*.txt")):
    #     dst_file = target_dir / src_file.name
    #     if dst_file.exists():
    #         import datetime as _dt
    #         ts = _dt.datetime.now().strftime("%Y%m%d%H%M%S")
    #         dst_file = target_dir / f"{src_file.stem}_{ts}{src_file.suffix}"
    #     shutil.move(str(src_file), str(dst_file))
    #     moved += 1
    #
    # print(f"\n{'='*60}")
    # print(f"DONE")
    # print(f"  Collected : {collected}/{num_rollouts}")
    # print(f"  Skipped   : {skipped}")
    # print(f"  Moved     : {moved} data files -> {target_dir}")
    # print(f"  Logs in   : {log_dir}")
    # print(f"{'='*60}")