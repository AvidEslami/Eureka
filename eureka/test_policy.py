import sys
import subprocess
import logging
from eureka import ISAAC_ROOT_DIR, EUREKA_ROOT_DIR
from utils.misc import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout, force=True)


def deploy_rollout(seed=1, task="ShadowHandSpin", suffix="", checkpoint=f"{ISAAC_ROOT_DIR}/checkpoints/EurekaPenSpinning.pth", capture_video=False):
    '''
    The goal of this function is to deploy a rollout of the policy on the environment and return the Fitness of the rollout.
        This fitness can be tentatively used to determine preference pairs.
    
    Manual Deploy Command Example:
    python train.py test=True headless=False force_render=True task=ShadowHandSpin checkpoint=checkpoints/EurekaPenSpinning.pth 
    '''
    
    rl_filepath = f"reward_code_eval_deploy_testing.txt"    
    with open(rl_filepath, 'w') as f:
        process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                    'hydra/output=subprocess',
                                    f'test=True', f'checkpoint={checkpoint}',
                                    f'task={task}{suffix}',
                                    f'headless={not capture_video}', f'capture_video={capture_video}', 'force_render=False', f'seed={seed}', 
                                    f'task.env.printNumSuccesses=True' ,
                                    ],
                                    stdout=f, stderr=f)
        success_score = block_until_finished_testing(rl_filepath, log_status=True)
        print(f"Process Completed. Success Score: {success_score}")
        return success_score  # Return the extracted success metric

def capture_rollout(seed=1, task="ShadowHandSpin", suffix="", checkpoint=f"{ISAAC_ROOT_DIR}/checkpoints/EurekaPenSpinning.pth", capture_video=False):
    '''
    The goal of this function is to deploy a rollout of the policy on the environment and save the list of states reached.
    
    Manual Deploy Command Example:
    python train.py test=True headless=False force_render=True task=ShadowHandSpin checkpoint=checkpoints/EurekaPenSpinning.pth 
    '''
    
    rl_filepath = f"reward_code_eval_deploy_testing.txt"    
    with open(rl_filepath, 'w') as f:
        process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                    'hydra/output=subprocess',
                                    f'test=True', f'checkpoint={checkpoint}',
                                    f'task={task}{suffix}',
                                    f'headless={not capture_video}', f'capture_video={capture_video}', 'force_render=False', f'seed={seed}', 
                                    f'task.env.printNumSuccesses=True',
                                    ],
                                    stdout=f, stderr=f)
        success_score = block_until_rollout_captured(rl_filepath, log_status=True, task_name=task)
        print(f"Process Completed. Success Score: {success_score}")
        return success_score  # Return the extracted success metric

if __name__ == "__main__":
    # task = "ShadowHand"
    # checkpoint = f"outputs/eureka/2025-01-28_02-15-55/policy-2025-01-28_04-57-03/runs/ShadowHandGPT-2025-01-28_04-57-03/nn/last_ShadowHandGPT_ep_20000.pth"
    # success = deploy_rollout()
    # success = deploy_rollout(task=task, checkpoint=checkpoint)
    # print(f"Final Success Score: {success}")

    capture_rollout()
    print(f"Finsihed Capturing Rollout")



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