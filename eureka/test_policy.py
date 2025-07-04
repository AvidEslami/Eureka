import sys
import subprocess
import logging
from eureka import ISAAC_ROOT_DIR, EUREKA_ROOT_DIR
from utils.misc import *
from pathlib import Path

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


def deploy_rollout(seed=1, task="ShadowHandSpin", suffix="", checkpoint=f"{ISAAC_ROOT_DIR}/checkpoints/EurekaPenSpinning.pth", capture_video=False):
    '''
    The goal of this function is to deploy a rollout of the policy on the environment and return the Fitness of the rollout.
        This fitness can be tentatively used to determine preference pairs.
    
    Manual Deploy Command Example:
    python train.py test=True headless=False force_render=True task=ShadowHandSpin checkpoint=checkpoints/EurekaPenSpinning.pth 
    '''
    
    rl_filepath = f"reward_code_eval_deploy_testing.txt"    
    with open(rl_filepath, 'w') as f:
        if task == "ShadowHand":
            process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                        'hydra/output=subprocess',
                                        f'test=True', f'checkpoint={checkpoint}',
                                        f'task={task}{suffix}',
                                        f'headless={not capture_video}', f'capture_video={capture_video}', 'force_render=False', f'seed={seed}', 
                                        f'task.env.printNumSuccesses=True' ,
                                        ],
                                        stdout=f, stderr=f)
        else:
            process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                        'hydra/output=subprocess',
                                        f'test=True', f'checkpoint={checkpoint}',
                                        f'task={task}{suffix}',
                                        f'headless={not capture_video}', f'capture_video={capture_video}', 'force_render=False', f'seed={seed}',                                        ],
                                        stdout=f, stderr=f)
        success_score = block_until_finished_testing(rl_filepath, log_status=True)
        # Terminate the process after capturing the rollout
        process.kill()
        print(f"Process Completed. Success Score: {success_score}")
        return success_score  # Return the extracted success metric

def capture_rollout(seed=2, task="ShadowHandSpin", suffix="", checkpoint=f"{ISAAC_ROOT_DIR}/checkpoints/EurekaPenSpinning.pth", capture_video=False, rl_filepath = "reward_code_eval_deploy_testing.txt", log_status=False):
    '''
    The goal of this function is to deploy a rollout of the policy on the environment and save the list of states reached.
    
    Manual Deploy Command Example:
    python train.py test=True headless=False force_render=True task=ShadowHandSpin checkpoint=checkpoints/EurekaPenSpinning.pth 
    '''
    if task == "ShadowHand":
        # rl_filepath = f"reward_code_eval_deploy_testing.txt"    
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                        'hydra/output=subprocess',
                                        f'test=True', f'checkpoint={checkpoint}',
                                        f'task={task}{suffix}',
                                        f'headless={not capture_video}', f'capture_video={capture_video}', 'force_render=True', f'seed={seed}', 
                                        f'task.env.printNumSuccesses=True'#, f'from_data=False'
                                        ],
                                        stdout=f, stderr=f)
            for success_value in monitor_direct_success(rl_filepath, process, log_status=log_status):
                    print(f"Current success: {success_value}")
                    if success_value == 2.0:
                        print("Success achieved!")
                        process.terminate()
                        try: 
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            print("Process did not terminate in time, killing it.")
                            process.kill()
                        stop_at_success = True
                        break

            stop_at_success = True
            # time.sleep(0.1)
            success_score = block_until_rollout_captured(rl_filepath, log_status=True, task_name=task, stop_at_success=stop_at_success, seed=seed)
            print(f"Process Completed. Success Score: {success_score}")
            return success_score  # Return the extracted success metric
    else:
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                        'hydra/output=subprocess',
                                        f'test=True', f'checkpoint={checkpoint}',
                                        f'task={task}{suffix}',
                                        f'headless={not capture_video}', f'capture_video={capture_video}', 'force_render=False', f'seed={seed}', 
                                        f'task.env.printSuccessStat=True' ,
                                        ],
                                        stdout=f, stderr=f)
            max_steps = 100
            average_success_score = block_until_rollout_captured(rl_filepath, log_status=True, task_name=task, stop_at_success=False, seed=seed, max_steps=max_steps)
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
        process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                    'hydra/output=subprocess',
                                    f'test=True', f'checkpoint={checkpoint}',
                                    f'task={task}{suffix}',
                                    f'headless={not capture_video}', f'capture_video={capture_video}', 'force_render=False', f'seed={seed}', 
                                    f'from_data=True', f'data_list={data_list_path}',
                                    f'task.env.printNumSuccesses=True'
                                    ],
                                    stdout=f, stderr=f)
        success_score = block_until_rollout_captured(rl_filepath, log_status=True, task_name=task)
        print(f"Process Completed. Success Score: {success_score}")
    return

def deploy_train(seed=1, task="ShadowHandSpin", suffix="", max_iterations=1000, checkpoint=f"{ISAAC_ROOT_DIR}/checkpoints/EurekaPenSpinning.pth", capture_video=False, rl_filepath="reward_code_eval_deploy_testing.txt"):
    '''
    The goal of this function is to deploy a rollout of the policy on the environment and return the Fitness of the rollout.
        This fitness can be tentatively used to determine preference pairs.
    
    Manual Deploy Command Example:
    python train.py test=True headless=False force_render=True task=ShadowHandSpin checkpoint=checkpoints/EurekaPenSpinning.pth 
    '''
    
    with open(rl_filepath, 'w') as f:
        process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                    'hydra/output=subprocess',
                                    f'task={task}{suffix}',
                                    f'headless={not capture_video}', f'capture_video={capture_video}', 'force_render=False', f'seed={seed}', 
                                    f'max_iterations={max_iterations}'
                                    ],
                                    stdout=f, stderr=f)
        success_score = block_until_training_finished(rl_filepath, log_status=True)
        print(f"Process Completed. Success Score: {success_score}")
        return success_score  # Return the extracted success metric

if __name__ == "__main__":
    # Train a ShadowHandDoorOpenOutward
    # deploy_train(seed=1, task="ShadowHandDoorOpenOutward", suffix="GPT", capture_video=False, max_iterations=3000)
    # exit()
    # Deploy a ShadowHandDoorOpenOutward rollout and capture videos
    policy_paths = ["/home/avidavid/Eureka/eureka/door_policies/ShadowHandDoorOpenOutwardGPT_successes_1210_1.00.pth",
                    "/home/avidavid/Eureka/eureka/door_policies/ShadowHandDoorOpenOutwardGPT_successes_360_0.50.pth",
                    "/home/avidavid/Eureka/eureka/door_policies/ShadowHandDoorOpenOutwardGPT_successes_100_0.00.pth"
    ]
    for policy_path in policy_paths:
        deploy_rollout(seed=1, task="ShadowHandDoorOpenOutward", suffix="GPT", checkpoint=policy_path, capture_video=True)
    exit()
    # Train a ShadowHand
    # deploy_train(seed=1,task="ShadowHand", suffix="GPT", capture_video=False, max_iterations=15000)
    # exit()
    # Train an Ant
    # deploy_train(seed=1, task="Ant", suffix="GPT", capture_video=False, max_iterations=2000, rl_filepath="reward_code_mlp_ant_new.txt")
    # exit()
    # Capture an Ant rollout
    # task = "Ant"
    # capture_rollout(task=task, checkpoint=f"{EUREKA_ROOT_DIR}/AntGPT.pth", capture_video=True)
    # capture_rollout(task=task, checkpoint=f"/home/avidavid/Eureka/eureka/policy-2025-06-12_15-33-14/runs/AntGPT-2025-06-12_15-33-15/nn/AntGPT_successes_500_7.19.pth", capture_video=True)
    # capture_rollout(task=task, seed=2, checkpoint=f"/home/avidavid/Eureka/eureka/outputs/eureka/2025-06-19_21-59-34/policy-2025-06-19_22-00-11/runs/AntGPT-2025-06-19_22-00-12/nn/AntGPT_successes_712_0.38.pth", capture_video=True)
    # exit()
    # # Capture extensive ant rollouts
    # checkpoints = []
    # task = "Ant"
    # checkpoints.append(f"../eureka/AntGPT.pth")
    # checkpoints.append(f"../eureka/policy-2025-06-12_15-33-14/runs/AntGPT-2025-06-12_15-33-15/nn/AntGPT_successes_500_7.19.pth")
    # checkpoints.append(f"../eureka/policy-2025-06-12_15-33-14/runs/AntGPT-2025-06-12_15-33-15/nn/AntGPT_successes_445_6.87.pth")
    # checkpoints.append(f"../eureka/policy-2025-06-12_15-33-14/runs/AntGPT-2025-06-12_15-33-15/nn/AntGPT_successes_395_6.58.pth")

    # curr_checkpoint = 0
    # while checkpoints:
    #     checkpoint = checkpoints.pop(0)
    #     curr_checkpoint += 1
    #     for i in range(1,4):
    #         print(f"Running Rollout {i} for checkpoint {curr_checkpoint}")
    #         current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #         policy_folder = checkpoint.split("/")[-3].split(".")[0]
    #         capture_rollout(task=task, checkpoint=checkpoint, seed=i, capture_video=False, rl_filepath=f"reward_code_{current_time}_{policy_folder},{i}.txt")
    #         time.sleep(2)
    # exit()


    # Capture extensive shadowhand rollouts
    checkpoints=[]
    task = "ShadowHand"
    checkpoints.append(f"outputs/eureka/2025-01-28_02-15-55/policy-2025-01-28_04-57-03/runs/ShadowHandGPT-2025-01-28_04-57-03/nn/last_ShadowHandGPT_ep_20000.pth")
    checkpoints.append(f"/home/avidavid/Eureka/eureka/policy-2025-03-20_23-58-54/runs/ShadowHandGPT-2025-03-20_23-58-55/nn/ShadowHandGPT.pth")
    checkpoints.append(f"/home/avidavid/Eureka/eureka/policy-2025-03-21_00-53-42/runs/ShadowHandGPT-2025-03-21_00-53-42/nn/last_ShadowHandGPT_ep_20000.pth")
    checkpoints.append(f"/home/avidavid/Eureka/eureka/policy-2025-03-21_02-03-38/runs/ShadowHandGPT-2025-03-21_02-03-39/nn/last_ShadowHandGPT_ep_2000.pth")
    # success = deploy_rollout(task=task, checkpoints=checkpoints)
    
    checkpoints.append(f"outputs/eureka/2025-01-28_02-15-55/policy-2025-01-28_02-16-22/runs/ShadowHandGPT-2025-01-28_02-16-22/nn/ShadowHandGPT.pth")
    checkpoints.append(f"outputs/eureka/2025-01-28_02-15-55/policy-2025-01-28_02-16-22/runs/ShadowHandGPT-2025-01-28_02-16-22/nn/last_ShadowHandGPT_ep_3000.pth")
    checkpoints.append(f"outputs/eureka/2025-01-28_02-15-55/policy-2025-01-28_03-35-01/runs/ShadowHandGPT-2025-01-28_03-35-01/nn/ShadowHandGPT.pth")
    checkpoints.append(f"outputs/eureka/2025-01-28_02-15-55/policy-2025-01-28_03-35-01/runs/ShadowHandGPT-2025-01-28_03-35-01/nn/last_ShadowHandGPT_ep_3000.pth")
    checkpoints.append(f"outputs/eureka/2025-01-28_02-15-55/policy-2025-01-28_04-57-03/runs/ShadowHandGPT-2025-01-28_04-57-03/nn/ShadowHandGPT.pth")
    
    # success = deploy_rollout()
    # print(f"Final Success Score: {success}")

    # task="Ant"
    # checkpoints.append(f"/home/avidavid/Eureka/eureka/outputs/eureka/2025-03-12_04-07-33/policy-2025-03-12_04-09-48/runs/AntGPT-2025-03-12_04-09-49/nn/AntGPT.pth"

    checkpoints.append("/home/avidavid/Eureka/eureka/policy-2025-05-01_19-35-14/runs/ShadowHandGPT-2025-05-01_19-35-15/nn/last_ShadowHandGPT_ep_5000.pth")
    checkpoints.append(f"/home/avidavid/Eureka/eureka/outputs/preferenced_eureka/2025-04-25_05-01-48/policy-2025-04-25_07-22-17/runs/ShadowHandGPT-2025-04-25_07-22-18/nn/ShadowHandGPT.pth")
    checkpoints.append(f"/home/avidavid/Eureka/eureka/policy-2025-05-02_04-56-29/runs/ShadowHandGPT-2025-05-02_04-56-30/nn/ShadowHandGPT.pth")

    checkpoints.append(f"/home/avidavid/Eureka/eureka/best_experiment_test/policy-2025-05-01_19-35-14/runs/ShadowHandGPT-2025-05-01_19-35-15/nn/ShadowHandGPT.pth")
    checkpoints.append(f"/home/avidavid/Eureka/eureka/best_experiment_test/policy-2025-05-02_03-29-38/runs/ShadowHandGPT-2025-05-02_03-29-39/nn/ShadowHandGPT.pth")
    checkpoints.append(f"/home/avidavid/Eureka/eureka/best_experiment_test/policy-2025-05-02_04-56-29/runs/ShadowHandGPT-2025-05-02_04-56-30/nn/ShadowHandGPT.pth")

    curr_checkpoint = 0
    while checkpoints:
        checkpoint = checkpoints.pop(0)
        curr_checkpoint += 1
        for i in range(1,4):
            print(f"Running Rollout {i} for checkpoint {curr_checkpoint}")
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            policy_folder = checkpoint.split("/")[-3].split(".")[0]
            capture_rollout(task=task, checkpoint=checkpoint,seed=i, capture_video=True, rl_filepath=f"reward_code_{current_time}_{policy_folder},{i}.txt")
            time.sleep(2)
    # print(f"Finsihed Capturing Rollout")

    # capture_reward_from_rollout(data_list_path="/home/avidavid/Eureka/eureka/ShadowHand_2025-02-28_01-49-17.txt", task=task, checkpoint=checkpoint)


    # {'params': {'seed': 42, 'algo': {'name': 'a2c_continuous'}, 'model': {'name': 'continuous_a2c_logstd'}, 
    #             'network': {'name': 'actor_critic', 'separate': False, 
    #                         'space': {'continuous': {'mu_activation': 'None', 'sigma_activation': 'None', 'mu_init': {'name': 'default'}, 'sigma_init': {'name': 'const_initializer', 'val': 0}, 'fixed_sigma': True
    #                                                  }
    #                                 }, 'mlp': {'units': [512, 512, 256, 128], 'activation': 'elu', 'd2rl': False, 'initializer': {'name': 'default'}, 'regularizer': {'name': 'None'}
    #                                            }
    #                         }, 'load_checkpoint': True, 'load_path': '/home/avidavid/Eureka/eureka/../isaacgymenvs/isaacgymenvs/checkpoints/EurekaPenSpinning.pth', 
    #             'config': {'name': 'ShadowHandSpin', 'full_experiment_name': None, 'env_name': 'rlgpu', 'multi_gpu': False, 'ppo': True, 'mixed_precision': False, 'normalize_input': True, 'normalize_value': True, 'value_bootstrap': True, 'num_actors': 1, 
    #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          'reward_shaper': {'scale_value': 0.01}, 'normalize_advantage': True, 'gamma': 0.99, 'tau': 0.95, 'learning_rate': 0.0005, 'lr_schedule': 'adaptive', 'schedule_type': 'standard', 'kl_threshold': 0.016, 'score_to_win': 100000, 'max_epochs': 20000, 'save_best_after': 100, 'save_frequency': 200, 'print_stats': True, 'grad_norm': 1.0, 'entropy_coef': 0.0, 'truncate_grads': True, 'e_clip': 0.2, 'horizon_length': 8, 'minibatch_size': 32768, 'mini_epochs': 5, 'critic_coef': 4, 'clip_value': True, 'seq_len': 4, 'bounds_loss_coef': 0.0001, 
    #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          'player': {'deterministic': True, 'games_num': 2000, 'print_stats': True}, 'log_dir': 'ShadowHandSpin-2025-02-02_18-36-32'}}}