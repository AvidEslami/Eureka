# VLM Querying Functions and Setup
import os
import json
import argparse
import requests
from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold
import time # Added this import for time.sleep
from typing import Dict, List, Optional

SELF_HOSTED_VLM = False
VERBOSE = False

if not SELF_HOSTED_VLM:
    api_key = os.getenv("GOOGLE_API_KEY")
    # Fallback: load from .env file at project root
    if not api_key:
        _env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        if os.path.exists(_env_path):
            with open(_env_path) as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line.startswith("GOOGLE_API_KEY="):
                        api_key = _line.split("=", 1)[1].strip()
                        break
    if not api_key:
        print("Error: GOOGLE_API_KEY not set and .env file not found.")
    else:
        client = genai.Client(api_key=api_key)


def get_task_description(task: str) -> Optional[str]:
    if task == "ShadowHandScissors":
        return "This class corresponds to the Scissors task. This environment involves two hands and scissors, we need to use two hands to open the scissors."
    elif task == "ShadowHandDoorOpenInward":
        return "Open the door using the two robotic hands, the door handles must first be grabbed, then pulled inwards in order to be opened."
    elif task == "ShadowHandDoorOpenOutward":
        return "Open the door using the two robotic hands, the door handles must first be grabbed, then pushed outwards in order to be opened."
    else:
        raise ValueError(f"Unknown task: {task}. Please provide a valid task description.")

def query_vlm_with_video(prompt: str, video_paths: List[str], verbose: bool=False) -> Optional[str]:
    uploaded_videos = [] # Initialize here for finally block

    # model = genai.GenerativeModel("gemini-2.5-pro") # This is correct for latest genai
    
    for i, path in enumerate(video_paths):
        if not os.path.exists(path):
            print(f"Video file {path} does not exist.")
            raise FileNotFoundError(f"Video file {path} does not exist.")
        
        if verbose:
            print(f"Uploading video: {path}")
        # These are the functions that cause the AttributeError if the package is old
        # video_file = genai.upload_file(path=path, display_name=f"video_{i+1}")
        vf = client.files.upload(file=path)

        # # Added timeout for robustness
        # start_time = time.time()
        # while video_file.state.name != "ACTIVE":
        #     if time.time() - start_time > 600: # 10 minute timeout for upload
        #         print(f"Video upload for {path} timed out.")
        #         raise Exception(f"Video upload for {path} timed out.")
            
        #     if verbose:
        #         print(f"Waiting for video {i+1} to be ready (Current state: {video_file.state.name})...")
        #     video_file = genai.get_file(video_file.name) # Check file status
        #     time.sleep(5) # Wait before checking again

        # if video_file.state.name == "FAILED":
        #     if verbose:
        #         print(f"Video upload failed for {path}.")
        #     raise Exception(f"Video upload failed for {path}.")

        # if verbose:
        #     print(f"Video {i+1} uploaded successfully. URI: {video_file.uri}") # Added URI for clarity
        # uploaded_videos.append(video_file)

        # New genai version
        # poll until ACTIVE (handle both .state or .state.name)
        start_time = time.time()
        while True:
            state = getattr(vf, "state", None)
            state_name = getattr(state, "name", state)
            if state_name == "ACTIVE":
                break
            if state_name == "FAILED":
                raise RuntimeError(f"Video upload failed for {path}.")
            if time.time() - start_time > 600:
                raise TimeoutError(f"Video upload for {path} timed out.")
            if verbose: print(f"Waiting for video {i+1} to be ready (Current state: {state_name})...")
            vf = client.files.get(name=vf.name)  # CHANGED
            time.sleep(5)

        if verbose: print(f"Video {i+1} uploaded successfully. URI: {vf.uri}")
        uploaded_videos.append(vf)

    if verbose:
        print("All videos uploaded successfully.")

    # request_content = [prompt] + uploaded_videos
    # response = model.generate_content(request_content, request_options={"timeout": 600})

    # # Define the safety settings to block none for all categories # Not needed anymore
    # safety_settings = {
    #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    #     # Add other categories if you are aware of them and wish to override their defaults
    #     # HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    # }

    schema = {
        "type": "object",
        "properties": {
            "winner": {
                "type": "string",
                "enum": ["1", "2"],  # <- strings, not ints
                "description": "1 = video 1 is better, 2 = video 2 is better"
            }
        },
        "required": ["winner"],
    }

    vf1 = uploaded_videos[0]  # First video file
    vf2 = uploaded_videos[1]  # Second video file

    # contents = [{
    #     "role": "user",
    #     "parts": [
    #         {"file_data": {"file_uri": vf1.uri, "mime_type": "video/mp4"}},
    #         {"text": f"This is Video 1: {os.path.basename(video_paths[0])}"},
    #         {"file_data": {"file_uri": vf2.uri, "mime_type": "video/mp4"}},
    #         {"text": f"This is Video 2: {os.path.basename(video_paths[1])}"},
    #         {"text": prompt},  # your main task instruction
    #     ],
    # }]

    # request_content = [prompt] + uploaded_videos

    # response = model.generate_content(
    query_count = 0
    responses = []
    while query_count < 2: # Retry up to 5 times
        try:
            if verbose:
                print(f"Query Count: {query_count + 1} / 2")
            # query_count += 1

            if query_count == 0:
                contents = [{
                    "role": "user",
                    "parts": [
                        {"file_data": {"file_uri": vf1.uri, "mime_type": "video/mp4"}},
                        {"text": f"This is Video 1: {os.path.basename(video_paths[0])}"},
                        {"file_data": {"file_uri": vf2.uri, "mime_type": "video/mp4"}},
                        {"text": f"This is Video 2: {os.path.basename(video_paths[1])}"},
                        {"text": prompt},  # your main task instruction
                    ],
                }]
            else:
                # Flip the order of the videos in the prompt
                contents = [{
                    "role": "user",
                    "parts": [
                        {"file_data": {"file_uri": vf2.uri, "mime_type": "video/mp4"}},
                        {"text": f"This is Video 1: {os.path.basename(video_paths[1])}"},
                        {"file_data": {"file_uri": vf1.uri, "mime_type": "video/mp4"}},
                        {"text": f"This is Video 2: {os.path.basename(video_paths[0])}"},
                        {"text": prompt},  # your main task instruction
                    ],
                }]
                

            response = client.models.generate_content(
                model="gemini-robotics-er-1.5-preview",  # Specify the model to use
                contents=contents,
                # safety_settings=safety_settings,
                config={
                    "temperature": 0.0,  # Adjust temperature for more randomness
                    "max_output_tokens": 5096,  # Limit the response length
                    "response_schema": schema,  # Use the defined schema for structured output
                    "response_mime_type": "application/json",  # Ensure the response is in JSON format
                },
            )

            # if verbose:
            #     print("Response: ", response)

            # Request with minimum randomness for consistency
            # response = model.generate_content(
            #     request_content,
            #     request_options={
            #         "temperature": 0.0,
            #         "max_output_tokens": 1024,
            #         "timeout": 600  # Timeout in seconds
            #     }
            # )

            if hasattr(response, "candidates") and response.candidates:
                content = getattr(response.candidates[0], "content", None)
                parts = getattr(content, "parts", None) if content is not None else None
                query_count += 1
                if parts:
                    for part in parts:
                        if hasattr(part, "text") and part.text:
                            if verbose:
                                print(f"Response part text: {part.text}")
                            try:
                                result = json.loads(part.text)
                                winner = int(result["winner"])
                                responses.append(winner)
                                if verbose:
                                    print(f"Parsed response: {winner}")
                                # return winner
                            except Exception as e:
                                if verbose:
                                    print(f"Failed to parse JSON from part text: {e}")
                else:
                    # In schema mode, there may be no text parts; use parsed instead
                    if getattr(response, "parsed", None) is not None:
                        winner = int(response.parsed["winner"])
                        responses.append(winner)
                        if verbose:
                            print(f"Parsed response (schema mode): {winner}")
                        # return winner
            else:
                if verbose:
                    print("No candidates found in response.")
                # return None

        except Exception as e:
            print(f"An error occurred while querying the VLM: {e}") 

    try:
        if verbose:
            print(f"Responses collected: {responses}")
        if responses and len(responses) >= 2:
            # Return the most common response if multiple were collected, if not just return 0
            if responses[0] != responses[1]:
                return responses[0]
            else:
                return 0
        else:
            print("Not enough responses collected, failing")
            return 5
    finally:
        # Now clean up the uploaded videos
        for video_file in uploaded_videos:
            try:
                # genai.delete_file(video_file.name)
                client.files.delete(name=video_file.name)  # CHANGED for new genai version
                if verbose:
                    print(f"Deleted temporary uploaded file: {video_file.name}")
            except Exception as e:
                print(f"Warning: Could not delete temporary file {video_file.name}: {e}")

            # if verbose:
            #     print("Response received from VLM.")
            # if response and response.text:
            #     return response.text
            # else:
            #     print("No response text received.")
            #     return None
        
    
        
        # # Print out Candidate info if verbose
        # if verbose:
        #     print("---- RAW RESPONSE DUMP ----")
        #     try:
        #         if response and hasattr(response, "candidates") and response.candidates:
        #             for idx, cand in enumerate(response.candidates):
        #                 print(f"\nCandidate {idx}:")
        #                 print(f"  Finish reason: {getattr(cand, 'finish_reason', None)}")
        #                 print(f"  Index: {getattr(cand, 'index', None)}")
        #                 print(f"  Safety ratings: {getattr(cand, 'safety_ratings', None)}")
        #                 content = getattr(cand, "content", None)
        #                 parts = getattr(content, "parts", None) if content is not None else None
        #                 if parts:
        #                     for p_idx, part in enumerate(parts):
        #                         print(f"    Part {p_idx}: {part}")
        #                 else:
        #                     # In schema mode, there may be no text parts at all
        #                     print("    No parts in content (likely structured-output only).")
        #         else:
        #             print("No candidates found in response.")
        #         # Bonus: also show structured/text if present
        #         if getattr(response, "parsed", None) is not None:
        #             print(f"parsed: {response.parsed}")
        #         elif getattr(response, "text", None):
        #             print(f"text: {response.text[:500]}")
        #     except Exception as debug_e:
        #         print(f"Error while dumping response: {debug_e}")


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
    task_prompt = "Evaluate the two trajectories demonstrated in the videos and decide which one is closer to the goal. The trajectories should be evaluated based the moment they are the most close to the task. If it were close to the goal, and moved away later, it should be judged by the moment it was close to the goal.Your answer should be [[1]], [[2]], or [[0]]. (1 corresponds to video 1, 2 corresponds to video 2). If the choice is arbitrary or the rollouts aren't discernable reply [[0]], you should respond with [[0]] if you don't see any meaningful progress in either video, only respond with [[1]] or [[2]] if one video is a lot better than the other. The goal:" + task_description
    response = query_vlm_with_video(task_prompt, video_paths, verbose=VERBOSE)

    if response is not None:
        if VERBOSE:
            print(f"Response from VLM: {response}")
        # Open a file at ./utils/vlm_response.txt and write 0 or 1 depending on whether [[1]] or [[2]] was found in the response
        with open("./utils/vlm_response.txt", "w") as f:
            # if "[[1]]" in response and "[[2]]" not in response:
            #     f.write("0")
            # elif "[[2]]" in response and "[[1]]" not in response:
            #     f.write("1")
            # else:
            #     print("Invalid response received from VLM. Check the logs for details.") # We'll set this up to requery
            #     f.write("5")
            if response == 1:
                f.write("1")
            elif response == 2:
                f.write("2")
            elif response == 0:
                f.write("0")
            else:
                print("Invalid response received from VLM. Check the logs for details.")
                f.write("5")
    else:
        print("No response received from VLM. Check the logs for details.")
        with open("./utils/vlm_response.txt", "w") as f:
            f.write("5")
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
    BOTTLE_TEST = False
    DOOR_INWARD_TEST = False    
    DOOR_INWARD_BENCHMARK = False
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
            exit()
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
    elif BOTTLE_TEST:
        task_description = "Remove the bottle cap from the bottle."
        test_prompt = "Which video does a better job at completing the task described by the following task description, answer with 1 or 2 surrounded by double square brackets, example: [[1]] or [[2]]: " + task_description
        test_video_paths = [
            "/home/avidavid/Eureka/eureka/dbottle_cap_videos/rl-video-step-0 copy.mp4",
            "/home/avidavid/Eureka/eureka/dbottle_cap_videos/rl-video-step-0.mp4"
        ]

        response = query_vlm_with_video(test_prompt, test_video_paths, verbose=True)
        print(f"\nResponse: {response}")
    elif DOOR_INWARD_TEST:
        task_description = "Open the door using the two robotic hands, the door must be pulled towards the camera to be opened."
        test_prompt = "Which video does a better job at completing the task described by the following task description, answer with 1 or 2 surrounded by double square brackets and favor partial progress, example: [[1]] or [[2]] (1 corresponds to video 1, 2 corresponds to video 2). If the choice is arbitrary or the rollouts aren't discernable reply [[0]], guessing the wrong preference is worse than saying that neither are better: " + task_description
        test_video_paths = [
            # "/home/avidavid/Eureka/eureka/door_inward_videos/rl-video-step-0 copy 5.mp4",
            "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/door_inward_videos/rl-video-step-0 copy 4.mp4",
            # "/home/avidavid/Eureka/eureka/door_inward_videos/rl-video-step-0 copy 3.mp4",
            "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/door_inward_videos/rl-video-step-0 copy 9.mp4"
        ]
        print("Task Description:", task_description)
        print("Video 1:", test_video_paths[0])
        print("Video 2:", test_video_paths[1])
        for i in range(5):
            response = query_vlm_with_video(test_prompt, test_video_paths, verbose=False)
            print(f"\nResponse: {response}")
        test_video_paths.reverse()
        for i in range(5):
            response = query_vlm_with_video(test_prompt, test_video_paths, verbose=False)
            print(f"\nResponse: {response}")
    elif DOOR_INWARD_BENCHMARK:

        task_description = "Open the doors using the two robotic hands, the door handles must first be grabbed, then pulled inwards in order to be opened."
        test_prompt = "Evaluate the two trajectories demonstrated in the videos and decide which one is closer to the goal. The trajectories should be evaluated based the moment they are the most close to the task. If it were close to the goal, and moved away later, it should be judged by the moment it was close to the goal.Your answer should be [[1]], [[2]], or [[0]]. (1 corresponds to video 1, 2 corresponds to video 2). The goal:" + task_description
        # test_video_paths = [
        #     # "/home/avidavid/Eureka/eureka/door_inward_videos/rl-video-step-0 copy 5.mp4",
        #     "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0 copy 4.mp4",
        #     "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0 copy 5.mp4",
        #     "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0 copy 9.mp4",
        #     "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0 copy 2.mp4",
        #     "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0 copy 7.mp4",
        #     "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0 copy 3.mp4",
        #     "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0 copy 10.mp4"
        # ]

        test_video_paths = [
            # "/home/avidavid/Eureka/eureka/door_inward_videos/rl-video-step-0 copy 5.mp4",
            "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0_copy_4.mp4",
            "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0_copy_5.mp4",
            "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0_copy_9.mp4",
            "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0_copy_2.mp4",
            "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0_copy_7.mp4",
            "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0_copy_3.mp4",
            "/home/avidavid/Eureka/eureka/door_inward_videos_cliped/rl-video-step-0_copy_10.mp4"
        ]

        successes = 0
        total = 0
        # for i in range(len(test_video_paths)):
        #     for j in range(i, len(test_video_paths)):
        #         if i == j:
        #             continue
        #         # print(f"\nTesting with videos: {test_video_paths[i]} and {test_video_paths[j]}")
        #         response = query_vlm_with_video(test_prompt, [test_video_paths[i], test_video_paths[j]], verbose=False)
        #         print("Response:", response)
        #         if i < j:
        #             # if "[[1]]" in response and "[[2]]" not in response:
        #             #     print(f"Video {i+1} is correctly preferred over Video {j+1}.")
        #             #     successes += 1
        #             # elif "[[2]]" in response and "[[1]]" not in response:
        #             #     print(f"Video {j+1} is incorrectly preferred over Video {i+1}.")
        #             # elif "[[0]]" in response:
        #             #     print(f"No preference found for {i+1} and {j+1}, response was [[0]].")
        #             if response == 1:
        #                 print(f"Video {i+1} is correctly preferred over Video {j+1}.")
        #                 successes += 1
        #             elif response == 2:
        #                 print(f"Video {j+1} is incorrectly preferred over Video {i+1}.")
        #             elif response == 0:
        #                 print(f"No preference found for {i+1} and {j+1}, response was [[0]].")
        #                 if (j - j) < 3:
        #                     print("This is a close call, VLM is likely confused, this is expected for these two videos.")
        #                     successes += 1
        #                 else:
        #                     print("This is unexpected, VLM should be able to discern these two videos.")
        #             else:
        #                 print("Invalid response received from VLM.")
        #                 print(f"Response: {response}")
        #             total += 1
                
        #         print("Benchmark Accuracy:", successes / total * 100)
        #         # print(f"\nResponse: {response}")
        # Now test with the order of the videos sent reversed
        for i in range(len(test_video_paths)):
            for j in range(i, len(test_video_paths)):
                if i == j:
                    continue
                # print(f"\nTesting with videos: {test_video_paths[j]} and {test_video_paths[i]}")
                response = query_vlm_with_video(test_prompt, [test_video_paths[j], test_video_paths[i]], verbose=False)
                print("Response:", response)
                if i < j:
                    # if "[[1]]" in response and "[[2]]" not in response:
                    #     print(f"Video {j+1} is correctly preferred over Video {i+1}.")
                    #     successes += 1
                    # elif "[[2]]" in response and "[[1]]" not in response:
                    #     print(f"Video {i+1} is incorrectly preferred over Video {j+1}.")
                    # elif "[[0]]" in response:
                    #     print(f"No preference found for {j+1} and {i+1}, response was [[0]].")
                    if response == 2:
                        print(f"Video {j+1} is correctly preferred over Video {i+1}.")
                        successes += 1
                    elif response == 1:
                        print(f"Video {j+1} is incorrectly preferred over Video {i+1}.")
                    elif response == 0:
                        print(f"No preference found for {j+1} and {i+1}, response was [[0]].")
                        if (j - j) < 3:
                            print("This is a close call, VLM is likely confused, this is expected for these two videos.")
                            successes += 1
                        else:
                            print("This is unexpected, VLM should be able to discern these two videos.")
                    else:
                        print("Invalid response received from VLM.")
                        print(f"Response: {response}")
                    total += 1
                
                print("Benchmark Accuracy:", successes / total * 100)