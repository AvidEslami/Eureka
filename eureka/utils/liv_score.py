# LIV Querying Functions and Setup
import os
import cv2
import argparse
import requests
import time # Added this import for time.sleep
from typing import Dict, List, Optional
import clip 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch 
import torchvision.transforms as T
from PIL import Image 
from liv import load_liv

VERBOSE = False

def get_short_task_description(task: str) -> Optional[str]:
    if task == "ShadowHandBottleCap":
        return "remove cap"
    else:
        raise ValueError(f"Unknown task: {task}. Please provide a valid task description.")

def query_liv_with_video(task_description: str, video_paths: List[str], verbose: bool=False) -> Optional[str]:
    try:
        # First we need to convert the video to a series of jpg images
        video_dir = "/home/avidavid/Eureka/eureka/utils/video_frames"

        # Delete video_dir if it exists then create it again
        if os.path.exists(video_dir):
            for file in os.listdir(video_dir):
                file_path = os.path.join(video_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(video_dir)  # Remove the directory itself
        os.makedirs(video_dir, exist_ok=True)

        frame_count = convert_video_to_jpegs(video_paths[0], video_dir)

        # Load model
        model = load_liv()
        model.eval()
        transform = T.Compose([T.ToTensor()])

        imgs = []
        imgs_tensor = []
        start_frame = -1
        end_frame = frame_count - 1
        for index in range(start_frame, end_frame):
            img = Image.open(f"{video_dir}/frame_0000{index+1:06}.jpg")
            imgs.append(img)
            imgs_tensor.append(transform(img))
        imgs_tensor = torch.stack(imgs_tensor)
        with torch.no_grad():
            embeddings = model(input=imgs_tensor.cuda(), modality="vision")
            token = clip.tokenize([task_description])
            goal_embedding_text = model(input=token, modality="text")
            goal_embedding_text = goal_embedding_text[0] 

        distances_cur_text = [] 
        for t in range(embeddings.shape[0]):
            cur_embedding = embeddings[t]
            cur_distance_text = - model.module.sim(goal_embedding_text, cur_embedding).detach().cpu().numpy()

            distances_cur_text.append(cur_distance_text)

        # distances_cur_text = np.array(distances_cur_text)

        return distances_cur_text
    except Exception as e:
        print(f"An error occurred while querying LIV: {e}")
        return None

def convert_video_to_jpegs(video_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Construct the filename for the JPEG image
        filename = f"frame_{frame_count:010d}.jpg"
        filepath = os.path.join(output_dir, filename)

        # Save the frame as a JPEG file
        cv2.imwrite(filepath, frame)
        frame_count += 1

    cap.release()
    print(f"Converted {frame_count} frames to JPEGs in '{output_dir}'")
    return frame_count
    
if __name__ == "__main__":

    # # Operation, parse command line for arguments, example run command could look like
    # # python vlm.py <task> </path/to/video1.mp4> </path/to/video2.mp4">
    
    # # Start parsing args:
    # parser = argparse.ArgumentParser(description="Query LIV with a prompt and video files.")
    # task = parser.add_argument("task", type=str, help="Task description for the VLM.")
    # video_path = parser.add_argument("video_path", type=str, nargs='+', help="Path to video file to be uploaded.")
    # args = parser.parse_args()
    # task = args.task
    # video_paths = args.video_paths
    # if VERBOSE:
    #     print(f"Querying LIV with task: {task} and video: {video_paths}")

    # task_description = get_short_task_description(task)
    # response_array = query_liv_with_video(task_description, video_paths, verbose=VERBOSE)

    # if VERBOSE:
    #     print(f"Response from LIV: {response_array}")
    # # Open a file at ./utils/vlm_response.txt and write 0 or 1 depending on whether [[1]] or [[2]] was found in the response
    # with open("./utils/liv_response.txt", "w") as f:
    #     f.write(str(response_array.tolist()))

    # exit()

    CAP_TEST = True
    task = "ShadowHandBottleCap"
    video_path = "/home/avidavid/LIV/liv/examples/eureka_video_2/rl-video-step-0(2).mp4"
    task_description = get_short_task_description(task)
    response_array = query_liv_with_video(task_description, [video_path], verbose=VERBOSE)
    print(f"Response from LIV: {response_array}")