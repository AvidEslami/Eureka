import subprocess
import os
import json
import logging
import re
import time
import datetime

from utils.file_utils import find_files_with_substring, find_folders_with_substring, load_tensorboard_logs
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

def block_until_training_finished(rl_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the RL training has started before moving on
    while True:
        rl_log = file_to_string(rl_filepath)
        if "MAX EPOCHS NUM" in rl_log or "Traceback" in rl_log:
            if log_status and "MAX EPOCHS NUM" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully finished training!")
                return True
            if log_status and "Traceback" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
                return False
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
        elif "reward:" in rl_log:
            logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully tested!")
            # The average consecutive fitness is the number at the end of the third line from the end
            # max_success = float(rl_log.split('\n')[-3].split()[-1])
            return True
        # # Stop when training completes
        # if "MAX EPOCHS NUM!" in rl_log or "Process Completed" in rl_log:
        #     break

    # return float(rl_log.split('\n')[-3].split()[-1])
    return max_success

def block_until_rollout_finished(rl_filepath, log_status=False, iter_num=-1, response_id=-1):
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

def block_until_rollout_captured(rl_filepath, log_status=False, iter_num=-1, response_id=-1, task_name="task_name", stop_at_success=False, seed=0, max_steps=None):
    if task_name == "ShadowHand":
        # Ensure that the RL training has started before moving on
        max_success = -1
        tensorboard_dir = None
        while True:
            rl_log = file_to_string(rl_filepath)

            if stop_at_success == False:
                if "average reward:" in rl_log or "Traceback" in rl_log:
                    if "average reward:" in rl_log:
                        logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully tested!")
                        # The average consecutive fitness is the number at the end of the third line from the end
                        
                        # Find the line that starts with: 'Post-Reset average consecutive successes:' and extract the number that follows
                        for line in reversed(rl_log.split("\n")):
                            if line.startswith("Post-Reset average consecutive successes = "):
                                max_success = float(line.split("=")[-1].strip())
                                break

                        # Now go through the entire log and save all the observations
                        # Observations were printed as follows
            else:   
                '''
                Observation: [[x,y,z,...]]
                ...
                Observation: [[a,b,c,...]]
                ...
                Observation: [[d,e,f,...]]
                '''
                max_success = 0
                obs_list = []
                for line in rl_log.split("\n"):
                    if line.startswith("Observations:"):
                        obs_list.append(json.loads(line.split(":")[-1].strip()))
                    elif line.startswith("Direct average consecutive successes = 1"):
                        max_success = 1
                    elif line.startswith("Direct average consecutive successes = 2"):
                        max_success = 2
                        break
                    # Store the observations in a file for later use named with task_date_time.txt
                date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                obs_filepath = f"/home/avidavid/Eureka/eureka/auto_preference_data/{seed}_{task_name}_{date_time}.txt"
                with open(obs_filepath, 'w') as f:
                        # On the first line writ the successes
                    f.write(f"{max_success}\n")
                    for obs in obs_list:
                        f.write(f"{obs}\n")
                return max_success
                if log_status and "Traceback" in rl_log:
                    logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
                break
            
            # # Stop when training completes
            # if "MAX EPOCHS NUM!" in rl_log or "Process Completed" in rl_log:
            #     break

        # return float(rl_log.split('\n')[-3].split()[-1])
        return max_success
    elif task_name == "ShadowHandScissors":
        # Read until the line: post_reset average consecutive successes: _ shows up, then find the success from that line and write it to a file
        while True:
            rl_log = file_to_string(rl_filepath)
            if "reward: " in rl_log or "Traceback" in rl_log:
                if "reward: " in rl_log:
                    # The average consecutive fitness is the number at the end of the third line from the end
                    # max_success = float(rl_log.split('\n')[-2].split()[-1])

                    # Link to video file that corresponds to this
                    if seed == 0:
                        policy_paths = "/home/avidavid/Eureka/eureka"
                    else: # Running inside peureka uses seeds 1,2,3
                        policy_paths = "/home/avidavid/Eureka/eureka/outputs/preferenced_eureka"
                        # Inside policy_paths look for the folder with the newest date and time, folder names are formatted as <yyyy-mm-dd_hh-mm-ss>
                        run_folders = os.listdir(policy_paths)
                        if not run_folders:
                            logging.error(f"No run folders found in {policy_paths}")
                            return False
                        # Find the folder that has the most recent date and time
                        run_folders.sort(key=lambda x: os.path.getmtime(os.path.join(policy_paths, x)), reverse=True)
                        # The most recent run folder is the first one in the sorted list
                        most_recent_run_folder = run_folders[0]
                        policy_paths = os.path.join(policy_paths, most_recent_run_folder)
                    # Open policy_paths, in this folder there will be several folders named policy-<yyyy-mm-dd_hh-mm-ss>
                    # Find the folder that has the most recent date and time
                    policy_folders = find_folders_with_substring(policy_paths, "policy-")
                    if not policy_folders:
                        logging.error(f"No policy folders found in {policy_paths}")
                        return False
                    # Sort the folders by date and time
                    policy_folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    # Get the most recent folder
                    most_recent_policy_folder = policy_folders[0]
                    # Video file is at most_recent_policy_folder/videos/<some folder (only one exists)>/rl-video-step-0.mp4
                    video_file_path = os.path.join(most_recent_policy_folder, "videos")
                    video_folders = os.listdir(video_file_path)
                    if not video_folders:
                        logging.error(f"No video folders found in {video_file_path}")
                        return False
                    # Any folder inside videos will do, we just need the video file
                    video_file_path = os.path.join(video_file_path, video_folders[0], "rl-video-step-0.mp4")

                    # Tensors to Capture (Reference of code running in env):
                    # print(f"Object Pos: {self.object_pos.tolist()}")
                    # print(f"Object Rot: {self.object_rot.tolist()}")
                    # print(f"Goal Pos: {self.goal_pos.tolist()}")
                    # print(f"Goal Rot: {self.goal_rot.tolist()}")
                    # print(f"Scissors Right Handle Pos: {self.scissors_right_handle_pos.tolist()}")
                    # print(f"Scissors Left Handle Pos: {self.scissors_left_handle_pos.tolist()}")
                    # print(f"Object Dof Pos: {self.object_dof_pos.tolist()}")
                    # print(f"Left Hand Pos: {self.left_hand_pos.tolist()}")
                    # print(f"Right Hand Pos: {self.right_hand_pos.tolist()}")
                    # print(f"Right Hand Ff Pos: {self.right_hand_ff_pos.tolist()}")
                    # print(f"Right Hand Mf Pos: {self.right_hand_mf_pos.tolist()}")
                    # print(f"Right Hand Rf Pos: {self.right_hand_rf_pos.tolist()}")
                    # print(f"Right Hand Lf Pos: {self.right_hand_lf_pos.tolist()}")
                    # print(f"Right Hand Th Pos: {self.right_hand_th_pos.tolist()}")
                    # print(f"Left Hand Ff Pos: {self.left_hand_ff_pos.tolist()}")
                    # print(f"Left Hand Mf Pos: {self.left_hand_mf_pos.tolist()}")
                    # print(f"Left Hand Rf Pos: {self.left_hand_rf_pos.tolist()}")
                    # print(f"Left Hand Lf Pos: {self.left_hand_lf_pos.tolist()}")
                    # print(f"Left Hand Th Pos: {self.left_hand_th_pos.tolist()}")

                    # Iterate through the log and keep track of every line'
                    object_pos = []
                    object_rot = []
                    goal_pos = []
                    goal_rot = []
                    scissors_right_handle_pos = []
                    scissors_left_handle_pos = []
                    object_dof_pos = []
                    left_hand_pos = []
                    right_hand_pos = []
                    right_hand_ff_pos = []
                    right_hand_mf_pos = []
                    right_hand_rf_pos = []
                    right_hand_lf_pos = []
                    right_hand_th_pos = []
                    left_hand_ff_pos = []
                    left_hand_mf_pos = []
                    left_hand_rf_pos = []
                    left_hand_lf_pos = []
                    left_hand_th_pos = []

                    for line in rl_log.split("\n"):
                        if line.startswith("Object Pos:"):
                            object_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Object Rot:"):
                            object_rot.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Goal Pos:"):
                            goal_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Goal Rot:"):
                            goal_rot.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Scissors Right Handle Pos:"):
                            scissors_right_handle_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Scissors Left Handle Pos:"):
                            scissors_left_handle_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Object Dof Pos:"):
                            object_dof_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Left Hand Pos:"):
                            left_hand_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Right Hand Pos:"):
                            right_hand_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Right Hand Ff Pos:"):
                            right_hand_ff_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Right Hand Mf Pos:"):
                            right_hand_mf_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Right Hand Rf Pos:"):
                            right_hand_rf_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Right Hand Lf Pos:"):
                            right_hand_lf_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Right Hand Th Pos:"):
                            right_hand_th_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Left Hand Ff Pos:"):
                            left_hand_ff_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Left Hand Mf Pos:"):
                            left_hand_mf_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Left Hand Rf Pos:"):
                            left_hand_rf_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Left Hand Lf Pos:"):
                            left_hand_lf_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Left Hand Th Pos:"):
                            left_hand_th_pos.append(json.loads(line.split(":")[-1].strip()))
                    # Store all the tensors in a file for later use named with task_date_time.txt
                    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    success_filepath = f"/home/avidavid/Eureka/eureka/auto_preference_data/{seed}_{task_name}_{date_time}.txt"
                    with open(success_filepath, 'w') as f:
                        f.write(f"{video_file_path}\n")
                        # f.write(f"Max Success: {max_success}\n")
                        f.write("Object Pos:\n")
                        for pos in object_pos:
                            f.write(f"{pos}\n")
                        f.write("Object Rot:\n")
                        for rot in object_rot:
                            f.write(f"{rot}\n")
                        f.write("Goal Pos:\n")
                        for pos in goal_pos:
                            f.write(f"{pos}\n")
                        f.write("Goal Rot:\n")
                        for rot in goal_rot:
                            f.write(f"{rot}\n")
                        f.write("Scissors Right Handle Pos:\n")
                        for pos in scissors_right_handle_pos:
                            f.write(f"{pos}\n")
                        f.write("Scissors Left Handle Pos:\n")
                        for pos in scissors_left_handle_pos:
                            f.write(f"{pos}\n")
                        f.write("Object Dof Pos:\n")
                        for pos in object_dof_pos:
                            f.write(f"{pos}\n")
                        f.write("Left Hand Pos:\n")
                        for pos in left_hand_pos:
                            f.write(f"{pos}\n")
                        f.write("Right Hand Pos:\n")
                        for pos in right_hand_pos:
                            f.write(f"{pos}\n")
                        f.write("Right Hand Ff Pos:\n")
                        for pos in right_hand_ff_pos:
                            f.write(f"{pos}\n")
                        f.write("Right Hand Mf Pos:\n")
                        for pos in right_hand_mf_pos:
                            f.write(f"{pos}\n")
                        f.write("Right Hand Rf Pos:\n")
                        for pos in right_hand_rf_pos:
                            f.write(f"{pos}\n")
                        f.write("Right Hand Lf Pos:\n")
                        for pos in right_hand_lf_pos:
                            f.write(f"{pos}\n")
                        f.write("Right Hand Th Pos:\n")
                        for pos in right_hand_th_pos:
                            f.write(f"{pos}\n")
                        f.write("Left Hand Ff Pos:\n")
                        for pos in left_hand_ff_pos:
                            f.write(f"{pos}\n")
                        f.write("Left Hand Mf Pos:\n")
                        for pos in left_hand_mf_pos:
                            f.write(f"{pos}\n")
                        f.write("Left Hand Rf Pos:\n")
                        for pos in left_hand_rf_pos:
                            f.write(f"{pos}\n")
                        f.write("Left Hand Lf Pos:\n")
                        for pos in left_hand_lf_pos:
                            f.write(f"{pos}\n")
                        f.write("Left Hand Th Pos:\n")
                        for pos in left_hand_th_pos:
                            f.write(f"{pos}\n")
                    logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully tested!")

                    return True
                if log_status and "Traceback" in rl_log:
                    logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
                break
    else:
        # Read until the line: reward:  _ shows up, then find the average success from previous lines and write to a file the root_states, and potentials
        while True:
            rl_log = file_to_string(rl_filepath)
            # Count the number of times "Consecutive successes:" appears in the log
            success_count = rl_log.count("Consecutive successes:")
            if success_count >= max_steps or "reward: " in rl_log or "Traceback" in rl_log:
                if success_count >= max_steps or "reward: " in rl_log:
                    logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully tested!")
                    # Iterate through the log and keep track of every line that started with 'Consecutive successes:'
                    consecutive_successes = []
                    root_states = []
                    potentials = []
                    prev_potentials = []
                    actions = []
                    # dof_pos = []
                    dof_vel = []
                    for line in rl_log.split("\n"):
                        if line.startswith("Consecutive successes:"):
                            consecutive_successes.append(float(line.split(":")[-1].strip()))
                        elif line.startswith("Root States:"):
                            root_states.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Potentials:"):
                            potentials.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Previous Potentials:"):
                            prev_potentials.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Actions:"):
                            actions.append(json.loads(line.split(":")[-1].strip()))
                        # elif line.startswith("Dof Pos:"):
                        #     # This is a line that contains the action taken
                        #     dof_pos.append(json.loads(line.split(":")[-1].strip()))
                        elif line.startswith("Dof Vel:"):
                            # This is a line that contains the action taken
                            dof_vel.append(json.loads(line.split(":")[-1].strip()))
                    
                    if consecutive_successes:
                        mean_success = sum(consecutive_successes) / len(consecutive_successes)
                        logging.info(f"Mean consecutive successes: {mean_success}")

                        # Store success, root_states, and potentials in a file
                        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        success_filepath = f"/home/avidavid/Eureka/eureka/auto_preference_data/{seed}_{task_name}_{date_time}.txt"
                        with open(success_filepath, 'w') as f:
                            f.write(f"Mean Success: {mean_success}\n")
                            f.write("Root States:\n")
                            for state in root_states:
                                f.write(f"{state}\n")
                            f.write("Potentials:\n")
                            for potential in potentials:
                                f.write(f"{potential}\n")
                            f.write("Previous Potentials:\n")
                            for prev_potential in prev_potentials:
                                f.write(f"{prev_potential}\n")
                            f.write("Actions:\n")
                            for action in actions:
                                f.write(f"{action}\n")
                            f.write("Dof Vel:\n")
                            for vel in dof_vel:
                                f.write(f"{vel}\n")
                                

                        return mean_success
                if log_status and "Traceback" in rl_log:
                    logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
                break
def monitor_direct_success(rl_filepath, process, log_status=False, interval=0.1):
    """
    Monitors the specified file and yields the success value each time 
    "Direct average consecutive successes =" appears in the log.

    Args:
        rl_filepath: Path to the log file
        process: subprocess.Popen object to monitor for termination
        log_status: Whether to log status updates
        interval: How often to check the file (in seconds)

    Yields:
        float: The success value after "Direct average consecutive successes ="
    """
    import time
    import os
    import re
    import logging

    if log_status:
        logging.info(f"Starting to monitor {rl_filepath} for success values")

    last_file_size = 0
    last_reported_value = None
    success_pattern = re.compile(r"Direct average consecutive successes\s*=\s*([0-9.]+)")

    while True:
        file_exists = os.path.exists(rl_filepath)
        file_grew = False

        if file_exists:
            current_size = os.path.getsize(rl_filepath)

            if current_size > last_file_size:
                with open(rl_filepath, 'r') as f:
                    f.seek(last_file_size)
                    new_content = f.read()
                    last_file_size = current_size
                    file_grew = True

                    matches = success_pattern.findall(new_content)
                    if matches:
                        latest_value = float(matches[-1])
                        if latest_value != last_reported_value:
                            if log_status:
                                logging.info(f"Success value: {latest_value}")
                            last_reported_value = latest_value
                            yield latest_value

        # If no new data AND process has exited, break the loop
        if not file_grew and process.poll() is not None:
            if log_status:
                logging.info("Process ended and no new data in log. Exiting monitor.")
            break

        time.sleep(interval)
        

if __name__ == "__main__":
    print(get_freest_gpu())