#!/usr/bin/env python3
"""
Build global preference rankings for rollouts using gemini-robotics-er-1.5-preview
with full video upload and dynamic thinking.

Uses binary insertion with transitivity to minimize VLM queries:
- If A > B and B > C, then A > C (no need to compare A and C directly)

By default, loads existing rankings and only inserts new rollouts.

Usage:
    python redo_comparisons.py [--verbose] [--fresh]
"""

import os
import sys
import json
import time
import argparse
from typing import List, Tuple, Optional, Dict

from google import genai
from google.genai import types

# Configuration
DATA_FOLDER = "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/auto_preference_data"
MODEL_NAME = "gemini-robotics-er-1.5-preview"

TASK_DESCRIPTION = "Open the doors using the two robotic hands, the door handles must first be grabbed, then pulled inwards in order to be opened."

SYSTEM_INSTRUCTION = """You are an expert robotics evaluator analyzing two videos of a robotic manipulation task.
EVALUATION CRITERIA:
1. **Grasp Quality**: Are the door handles grasped firmly by both hands?
2. **Correct Motion**: Is the door being pulled inward (toward camera)?
3. **Door Opening**: How far does the door open?
4. **Peak Performance**: Judge by the BEST moment achieved in each video
5. **Sustained Progress**: Brief success is still better than no success

SUCCESS INDICATORS (in priority order):
- CRITICAL: Both hands successfully grasp door handles
- HIGH: Door moves inward (any amount)
- MEDIUM: Door opens 15+ degrees
- IDEAL: Door opens 45+ degrees and stays open

COMMON FAILURES TO RECOGNIZE:
- Hands miss or just touch handles (without grasping)
- Pushing instead of pulling
- One hand grasps but other doesn't
- Door opens briefly then closes

INSTRUCTIONS:
- Watch both videos completely
- Identify the peak moment of progress in each
- Compare peak achievements, not final states
- Recognize partial progress (touching < grasping < pulling < opening)
- If both videos show similar progress or both fail completely, choose 0
Return strictly one token: [1] if the FIRST video better completes the task, [2] if the SECOND video does, or [0] if they are similar/both fail.
Do not explain. Do not include any other characters."""

# Initialize client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    sys.exit(1)

client = genai.Client(api_key=api_key)


def load_rollouts(data_folder: str) -> Dict[str, str]:
    """
    Load all rollout files and extract their video paths.
    
    Returns:
        Dict mapping filename -> video_path
    """
    rollouts = {}
    for filename in os.listdir(data_folder):
        if not filename.endswith(".txt"):
            continue
        if filename == "preference_rankings.txt":
            continue
        
        filepath = os.path.join(data_folder, filename)
        try:
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                # First line should be a video path (starts with /)
                if first_line.startswith("/"):
                    rollouts[filename] = first_line
                else:
                    # If first line is a score, skip this file
                    print(f"  Skipping {filename}: first line is not a video path")
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
    
    return rollouts


def upload_video(video_path: str, verbose: bool = False):
    """Upload a video file and wait for it to become ACTIVE."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")
    
    if verbose:
        print(f"    Uploading: {os.path.basename(video_path)}")
    
    vf = client.files.upload(file=video_path)
    
    # Poll until ACTIVE
    start_time = time.time()
    while True:
        state = getattr(vf, "state", None)
        state_name = getattr(state, "name", state)
        if state_name == "ACTIVE":
            break
        if state_name == "FAILED":
            raise RuntimeError(f"Video upload failed for {video_path}.")
        if time.time() - start_time > 600:
            raise TimeoutError(f"Video upload for {video_path} timed out.")
        if verbose:
            print(f"      Waiting... (state: {state_name})")
        vf = client.files.get(name=vf.name)
        time.sleep(5)
    
    if verbose:
        print(f"      Upload complete.")
    
    return vf


def delete_video(vf, verbose: bool = False):
    """Delete an uploaded video file."""
    try:
        client.files.delete(name=vf.name)
        if verbose:
            print(f"    Deleted: {vf.name}")
    except Exception as e:
        if verbose:
            print(f"    Warning: Could not delete {vf.name}: {e}")


def parse_choice(text: str) -> Optional[int]:
    """Parse the model's response to extract 0, 1, or 2."""
    if not text:
        return None
    s = text.strip().lower()
    if "[1]" in s or "[[1]]" in s or s == "1":
        return 1
    if "[2]" in s or "[[2]]" in s or s == "2":
        return 2
    if "[0]" in s or "[[0]]" in s or s == "0":
        return 0
    # Fallback: find first digit 0, 1, or 2
    import re
    m = re.search(r"\b([012])\b", s)
    if m:
        return int(m.group(1))
    return None


def compare_videos(
    video1_path: str,
    video2_path: str,
    task_description: str,
    verbose: bool = False,
) -> Tuple[int, str]:
    """
    Compare two videos using gemini-robotics-er-1.5-preview with full video upload.
    
    Returns:
        Tuple of (result, raw_response)
        result: 1 = video1 better, 2 = video2 better, 0 = tie/similar, -1 = error
    """
    uploaded_videos = []
    
    try:
        # Upload both videos
        vf1 = upload_video(video1_path, verbose)
        uploaded_videos.append(vf1)
        
        vf2 = upload_video(video2_path, verbose)
        uploaded_videos.append(vf2)
        
        # Build the prompt
        prompt = f"""Task description: {task_description}
Decide which video demonstrates better progress toward opening the door. Answer only [1], [2], or [0].
Video 1 follows:"""

        # Build contents with videos
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=vf1.uri, mime_type="video/mp4"),
                    types.Part.from_text(text="This is Video 1."),
                    types.Part.from_uri(file_uri=vf2.uri, mime_type="video/mp4"),
                    types.Part.from_text(text="This is Video 2."),
                    types.Part.from_text(text=prompt),
                ],
            )
        ]

        if verbose:
            print(f"    Querying model: {MODEL_NAME}")

        # Make the API call with dynamic thinking
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.0,
                max_output_tokens=8192,
                thinking_config=types.ThinkingConfig(thinking_budget=-1)  # Dynamic thinking
            ),
        )

        # Extract the response text
        raw_response = ""
        if hasattr(response, "text") and response.text:
            raw_response = response.text
        elif hasattr(response, "candidates") and response.candidates:
            content = getattr(response.candidates[0], "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                for part in parts:
                    if hasattr(part, "text") and part.text:
                        raw_response = part.text
                        break
        
        if verbose:
            print(f"    Raw response: {raw_response}")
        
        result = parse_choice(raw_response)
        if result is None:
            return -1, raw_response
        
        return result, raw_response
        
    except Exception as e:
        print(f"    Error: {e}")
        return -1, str(e)
    
    finally:
        # Clean up uploaded videos
        for vf in uploaded_videos:
            delete_video(vf, verbose)


def vlm_compare(video_path_a: str, video_path_b: str, verbose: bool = False) -> int:
    """
    Compare two rollouts via VLM.
    
    Returns:
        1 = A is better
        2 = B is better
        0 = tie
        5 = error
    """
    result, raw = compare_videos(video_path_a, video_path_b, TASK_DESCRIPTION, verbose)
    
    if result == -1:
        return 5  # Error
    return result


def save_rankings(data_folder: str, global_order: List[str], tied_pairs: List[Tuple[str, str]]):
    """Save the current rankings to preference_rankings.txt"""
    rankings_path = os.path.join(data_folder, "preference_rankings.txt")
    with open(rankings_path, 'w') as f:
        f.write(str(global_order))
        f.write("\n")
        f.write(str(tied_pairs))
    print(f"  Saved rankings to {rankings_path}")


def load_existing_rankings(data_folder: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Load existing rankings if they exist."""
    rankings_path = os.path.join(data_folder, "preference_rankings.txt")
    if os.path.exists(rankings_path):
        try:
            with open(rankings_path, 'r') as f:
                global_order = eval(f.readline().strip())
                tied_pairs = eval(f.readline().strip())
            return global_order, tied_pairs
        except Exception as e:
            print(f"  Warning: Could not load existing rankings: {e}")
    return [], []


def binary_insert(
    new_name: str,
    new_video_path: str,
    global_order: List[str],
    tied_pairs: List[Tuple[str, str]],
    rollouts: Dict[str, str],
    verbose: bool = False
) -> bool:
    """
    Insert a new rollout into the global order using binary search.
    Uses VLM comparison to determine placement.
    
    The global_order is sorted from WORST to BEST (weak -> strong).
    
    Returns:
        True if inserted successfully, False otherwise
    """
    if verbose:
        print(f"  Binary inserting: {new_name}")
    
    lo, hi = 0, len(global_order)
    
    while lo < hi:
        mid = (lo + hi) // 2
        existing_name = global_order[mid]
        
        if existing_name not in rollouts:
            print(f"    Warning: {existing_name} not in rollouts, skipping comparison")
            lo = mid + 1
            continue
        
        new_video = new_video_path
        existing_video = rollouts[existing_name]
        
        if verbose:
            print(f"    Comparing vs {existing_name} (position {mid})")
        
        # VLM compare: 1 = new wins, 2 = existing wins, 0 = tie
        result = vlm_compare(new_video, existing_video, verbose)
        
        if result == 5:
            print(f"    VLM error comparing {new_name} vs {existing_name}, skipping")
            return False
        
        if result == 1:
            # New is better than existing -> move right (toward better end)
            if verbose:
                print(f"    {new_name} > {existing_name}, moving right")
            lo = mid + 1
        elif result == 2:
            # Existing is better than new -> move left (toward worse end)
            if verbose:
                print(f"    {new_name} < {existing_name}, moving left")
            hi = mid
        else:
            # Tie (result == 0)
            if verbose:
                print(f"    {new_name} ~ {existing_name}, recording tie")
            tied_pairs.append((new_name, existing_name))
            # Insert next to the tied element
            global_order.insert(mid, new_name)
            return True
    
    # Insert at the determined position
    if lo >= len(global_order) or global_order[lo] != new_name:
        global_order.insert(lo, new_name)
        if verbose:
            print(f"    Inserted {new_name} at position {lo}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Build global preference rankings with binary insertion")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--fresh", action="store_true", help="Start fresh, ignoring existing rankings")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Building Global Preference Rankings")
    print("=" * 80)
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Model: {MODEL_NAME}")
    print(f"Thinking: Dynamic (budget=-1)")
    print("=" * 80)
    
    # Load rollouts
    print("\nLoading rollouts...")
    rollouts = load_rollouts(DATA_FOLDER)
    print(f"Found {len(rollouts)} rollouts with video paths")
    
    if len(rollouts) == 0:
        print("No rollouts found!")
        return 1
    
    # Load existing rankings (default behavior, use --fresh to start over)
    if args.fresh:
        print("\nStarting fresh (ignoring existing rankings)...")
        global_order = []
        tied_pairs = []
    else:
        print("\nLoading existing rankings...")
        global_order, tied_pairs = load_existing_rankings(DATA_FOLDER)
        print(f"  Loaded {len(global_order)} ranked rollouts, {len(tied_pairs)} tied pairs")
    
    # Sort rollouts by filename for consistent ordering
    rollout_names = sorted(rollouts.keys())
    
    # Find rollouts not yet in the global order
    to_insert = [name for name in rollout_names if name not in global_order]
    
    # Also check tied_pairs for already-placed items
    placed_via_ties = set()
    for tie in tied_pairs:
        placed_via_ties.add(tie[0])
        placed_via_ties.add(tie[1])
    
    to_insert = [name for name in to_insert if name not in placed_via_ties]
    
    print(f"\nRollouts to insert: {len(to_insert)}")
    print(f"Already ranked: {len(global_order)}")
    print("-" * 80)
    
    # Process each new rollout
    num_comparisons = 0
    errors = 0
    
    for idx, name in enumerate(to_insert):
        print(f"\n[{idx+1}/{len(to_insert)}] Processing: {name}")
        
        video_path = rollouts[name]
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"  Warning: Video not found: {video_path}")
            errors += 1
            continue
        
        # Binary insert
        success = binary_insert(
            new_name=name,
            new_video_path=video_path,
            global_order=global_order,
            tied_pairs=tied_pairs,
            rollouts=rollouts,
            verbose=args.verbose
        )
        
        if success:
            # Estimate comparisons: log2(n) for binary insertion
            import math
            num_comparisons += max(1, int(math.log2(max(1, len(global_order)))))
        else:
            errors += 1
        
        # Save after each insertion
        save_rankings(DATA_FOLDER, global_order, tied_pairs)
        
        # Small delay to avoid rate limiting
        time.sleep(1)
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total rollouts: {len(rollouts)}")
    print(f"Ranked rollouts: {len(global_order)}")
    print(f"Tied pairs: {len(tied_pairs)}")
    print(f"Errors: {errors}")
    print(f"Estimated VLM comparisons: {num_comparisons}")
    
    # Print ranking (best first)
    print("\nFinal Ranking (BEST to WORST):")
    print("-" * 40)
    for i, name in enumerate(reversed(global_order)):
        rank = i + 1
        # Check if this item has ties
        ties = [t[1] if t[0] == name else t[0] for t in tied_pairs if name in t]
        tie_str = f" (tied with: {', '.join(ties)})" if ties else ""
        print(f"  {rank}. {name}{tie_str}")
    
    # Save final results
    save_rankings(DATA_FOLDER, global_order, tied_pairs)
    
    # Also save detailed results as JSON
    results_path = os.path.join(DATA_FOLDER, "ranking_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model': MODEL_NAME,
            'thinking': 'dynamic',
            'num_rollouts': len(rollouts),
            'num_ranked': len(global_order),
            'num_tied_pairs': len(tied_pairs),
            'errors': errors,
            'global_order': global_order,  # worst to best
            'tied_pairs': tied_pairs,
        }, f, indent=2)
    print(f"\nSaved detailed results to: {results_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
