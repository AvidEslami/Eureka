import subprocess
import os
import json
import logging
import re
import time

from utils.file_utils import find_files_with_substring, load_tensorboard_logs
from utils.extract_task_code import file_to_string

def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def get_freest_gpu():
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    # Find GPU with most free memory
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])

    return freest_gpu['index']

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_training(rl_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the RL training has started before moving on
    while True:
        rl_log = file_to_string(rl_filepath)
        if "fps step:" in rl_log or "Traceback" in rl_log:
            if log_status and "fps step:" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            if log_status and "Traceback" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break

def block_until_finished_testing(rl_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the RL training has started before moving on
    max_success = -1
    tensorboard_dir = None
    while True:
        rl_log = file_to_string(rl_filepath)

        # TENSORBOARDS ARE NOT GENERATED IN TEST MODE
        # if log_status:
        # for line in rl_log.split("\n"):
        #     if line.startswith("Tensorboard Directory:"):
        #         tensorboard_dir = line.split(":")[-1].strip()
        #         break
                    # for attempt in range(5):
                    #     if os.path.exists(tensorboard_dir):
                    #         try:
                    #             tensorboard_logs = load_tensorboard_logs(tensorboard_dir)
                    #             if "consecutive_success" in tensorboard_logs:
                    #                 max_success = max(max_success, max(tensorboard_logs["consecutive_success"]))
                    #                 logging.info(f"Iteration {iter_num}: Code Run {response_id} - Max Success: {max_success}")
                    #                 return max_success
                    #         except: # If tensorboard logs are not ready yet
                    #             time.sleep(2)
                    #             pass



        if "average reward:" in rl_log or "Traceback" in rl_log:
            if "average reward:" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully tested!")
                # The average consecutive fitness is the number at the end of the third line from the end
                max_success = float(rl_log.split('\n')[-3].split()[-1])
                return max_success
            if log_status and "Traceback" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break
        
        # # Stop when training completes
        # if "MAX EPOCHS NUM!" in rl_log or "Process Completed" in rl_log:
        #     break

    # return float(rl_log.split('\n')[-3].split()[-1])
    return max_success

if __name__ == "__main__":
    print(get_freest_gpu())