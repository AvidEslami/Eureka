# VLM Querying Functions and Setup
import os
import argparse
import requests
import google.generativeai as genai
import time # Added this import for time.sleep
from typing import Dict, List, Optional

SELF_HOSTED_VLM = False
VERBOSE = False

if not SELF_HOSTED_VLM:
    api_key = os.getenv("GOOGLE_API_KEY")
    # Added API key configuration here, outside the function for cleaner setup
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        # In a real application, you might want to exit or raise a more specific error
        # For now, will proceed, but subsequent API calls will likely fail.
    else:
        genai.configure(api_key=api_key)

def get_task_description(task: str) -> Optional[str]:
    if task == "ShadowHandScissors":
        return "This class corresponds to the Scissors task. This environment involves two hands and scissors, we need to use two hands to open the scissors."
    else:
        raise ValueError(f"Unknown task: {task}. Please provide a valid task description.")

def query_vlm_with_video(prompt: str, video_paths: List[str], verbose: bool=False) -> Optional[str]:
    uploaded_videos = [] # Initialize here for finally block
    try:
        model = genai.GenerativeModel("gemini-2.5-flash") # This is correct for latest genai
        
        for i, path in enumerate(video_paths):
            if not os.path.exists(path):
                print(f"Video file {path} does not exist.")
                raise FileNotFoundError(f"Video file {path} does not exist.")
            
            if verbose:
                print(f"Uploading video: {path}")
            # These are the functions that cause the AttributeError if the package is old
            video_file = genai.upload_file(path=path, display_name=f"video_{i+1}")

            # Added timeout for robustness
            start_time = time.time()
            while video_file.state.name != "ACTIVE":
                if time.time() - start_time > 600: # 10 minute timeout for upload
                    print(f"Video upload for {path} timed out.")
                    raise Exception(f"Video upload for {path} timed out.")
                
                if verbose:
                    print(f"Waiting for video {i+1} to be ready (Current state: {video_file.state.name})...")
                video_file = genai.get_file(video_file.name) # Check file status
                time.sleep(5) # Wait before checking again

            if video_file.state.name == "FAILED":
                if verbose:
                    print(f"Video upload failed for {path}.")
                raise Exception(f"Video upload failed for {path}.")

            if verbose:
                print(f"Video {i+1} uploaded successfully. URI: {video_file.uri}") # Added URI for clarity
            uploaded_videos.append(video_file)

        if verbose:
            print("All videos uploaded successfully.")

        request_content = [prompt] + uploaded_videos
        response = model.generate_content(request_content, request_options={"timeout": 600})

        if verbose:
            print("Response received from VLM.")
        if response and response.text:
            return response.text
        else:
            print("No response text received.")
            return None
        
    except Exception as e:
        print(f"An error occurred while querying the VLM: {e}")
        return None
    finally:
        # Ensure uploaded files are deleted to free up resources
        for video_file in uploaded_videos:
            try:
                genai.delete_file(video_file.name)
                if verbose:
                    print(f"Deleted temporary uploaded file: {video_file.name}")
            except Exception as e:
                print(f"Warning: Could not delete temporary file {video_file.name}: {e}")

if __name__ == "__main__":

    # Operation, parse command line for arguments, example run command could look like
    # python vlm.py <task> </path/to/video1.mp4> </path/to/video2.mp4">
    
    # Start parsing args:
    parser = argparse.ArgumentParser(description="Query a Video Language Model (VLM) with a prompt and video files.")
    task = parser.add_argument("task", type=str, help="Task description for the VLM.")
    video_paths = parser.add_argument("video_paths", type=str, nargs='+', help="Paths to video files to be uploaded.")
    args = parser.parse_args()
    task = args.task
    video_paths = args.video_paths
    if VERBOSE:
        print(f"Querying VLM with task: {task} and videos: {video_paths}")

    task_description = get_task_description(task)
    task_prompt = "Which video does a better job at completing the task described by the following task description, answer with 1 or 2 surrounded by double square brackets, example: [[1]] or [[2]]: " + task_description
    response = query_vlm_with_video(task_prompt, video_paths, verbose=VERBOSE)

    if response:
        if VERBOSE:
            print(f"Response from VLM: {response}")
        # Open a file at ./utils/vlm_response.txt and write 0 or 1 depending on whether [[1]] or [[2]] was found in the response
        with open("./utils/vlm_response.txt", "w") as f:
            if "[[1]]" in response and "[[2]]" not in response:
                f.write("0")
            elif "[[2]]" in response and "[[1]]" not in response:
                f.write("1")
            else:
                print("Invalid response received from VLM. Check the logs for details.") # We'll set this up to requery
                f.write("5")
    else:
        print("No response received from VLM. Check the logs for details.")
        f.write("5")  # Indicating no response received
    exit()

    # # Sanity test
    # test_prompt = "What is happening in this video, also what is the name of this video?"
    # # Ensure this path is absolutely correct and accessible
    # test_video_path = ["/home/avidavid/Eureka/eureka/policy-2025-06-03_03-42-14/videos/ShadowHand_2025-06-03_03-42-15/rl-video-step-0.mp4"]
    
    # print(f"Attempting to query VLM with video: {test_video_path[0]}")
    # response = query_vlm_with_video(test_prompt, test_video_path, verbose=True)
    # print(f"\nResponse: {response}")
    DOOR_TEST = False
    SCISSOR_TEST = False
    if DOOR_TEST:
        # VLM Preference Sanity Test
        # task_description = "This class corresponds to the DoorOpenOutward task. This environment require a opened door  to be closed and the door can only be pushed outward or initially open inward. Both these two  environments only need to do the push behavior, so it is relatively simple"
        task_description = "This task requires that two doors be opened fully outwards."
        test_prompt = "Which video does a better job at completing the task described by the following task description, answer with 1 or 2 surrounded by double square brackets, example: [[1]] or [[2]]: " + task_description
        test_video_paths = [
            # "/home/avidavid/Eureka/eureka/door_videos/rl-video-step-best.mp4",
            "/home/avidavid/Eureka/eureka/door_videos/rl-video-step-middle.mp4",
            "/home/avidavid/Eureka/eureka/door_videos/rl-video-step-worst.mp4",
        ]
        # print(f"Attempting to query VLM with videos: {test_video_paths[0]} and {test_video_paths[1]}")

        score = 0
        test_count = 10
        for i in range(test_count):
            print(f"\nTest {i+1}/{test_count}")
            # time.sleep(1)  # Added delay between tests for clarity
            response = query_vlm_with_video(test_prompt, test_video_paths)
            if "[[1]]" in response and "[[2]]" not in response:
                score += 1
        # Print the score after test_count tests, then flip the order of the videos and try another test_count of tests
        print(f"\nScore after {test_count} tests: {score}/{test_count}")
        # Wait 1 minute to avoid rate limiting issues, api allows 10 requests per minute
        time.sleep(45)

        test_video_paths.reverse()  # Flip the order of the videos
        # print(f"Attempting to query VLM with videos: {test_video_paths[0]} and {test_video_paths[1]}")
        for i in range(test_count):
            print(f"\nTest {i+1}/{test_count}")
            # time.sleep(1)  # Added delay between tests for clarity
            response = query_vlm_with_video(test_prompt, test_video_paths)
            if "[[2]]" in response and "[[1]]" not in response:
                score += 1
            
        print(f"\nFinal Score after {test_count * 2} tests: {score}/{test_count * 2}")
        print(f"Preference Accuracy: {score / (test_count * 2) * 100:.2f}%")

        # print(f"\nResponse: {response}")
    elif SCISSOR_TEST:
        task_description = "This class corresponds to the Scissors task. This environment involves two hands and scissors,  we need to use two hands to open the scissors"
        test_prompt = "Which video does a better job at completing the task described by the following task description, answer with 1 or 2 surrounded by double square brackets, example: [[1]] or [[2]]: " + task_description
        test_video_paths = [
            "/home/avidavid/Eureka/eureka/dscissor_videos/rl-video-step-y.mp4",
            "/home/avidavid/Eureka/eureka/dscissor_videos/rl-video-step-x.mp4",
        ]

        score = 0
        test_count = 3
        for i in range(test_count):
            print(f"\nTest {i+1}/{test_count}")
            # time.sleep(1)  # Added delay between tests for clarity
            response = query_vlm_with_video(test_prompt, test_video_paths)
            if "[[1]]" in response and "[[2]]" not in response:
                score += 1
        # Print the score after test_count tests, then flip the order of the videos and try another test_count of tests
        print(f"\nScore after {test_count} tests: {score}/{test_count}")
        # Wait 1 minute to avoid rate limiting issues, api allows 10 requests per minute
        if test_count >= 10:
            time.sleep(45)
        test_video_paths.reverse()
        # print(f"Attempting to query VLM with videos: {test_video_paths[0]} and {test_video_paths[1]}")
        for i in range(test_count):
            print(f"\nTest {i+1}/{test_count}")
            # time.sleep(1)  # Added delay between tests for clarity
            response = query_vlm_with_video(test_prompt, test_video_paths)
            if "[[2]]" in response and "[[1]]" not in response:
                score += 1
        print(f"\nFinal Score after {test_count * 2} tests: {score}/{test_count * 2}")
        print(f"Preference Accuracy: {score / (test_count * 2) * 100:.2f}%")