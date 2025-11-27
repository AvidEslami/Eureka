import os
import re
import cv2
import sys
import time
import json
import base64
import random
import urllib.request as _urlreq
import urllib.error as _urlerr
from itertools import combinations
from typing import List, Tuple, Dict, Optional


def _read_env_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")
    return api_key


def _list_videos(folder: str) -> List[str]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
    if not files:
        raise RuntimeError(f"No video files found in folder: {folder}")
    files.sort()
    return [os.path.join(folder, f) for f in files]


def _extract_index_from_filename(path: str) -> Optional[int]:
    name = os.path.basename(path)
    # Extract the first integer found in filename
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def _load_video_capture(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def _sample_frame_indices(total_frames: int, max_frames: int) -> List[int]:
    if total_frames <= 0:
        return []
    if total_frames <= max_frames:
        return list(range(total_frames))
    step = total_frames / float(max_frames)
    return [int(i * step) for i in range(max_frames)]


def _resize_keep_aspect(img: "cv2.Mat", target_width: int) -> "cv2.Mat":
    h, w = img.shape[:2]
    if w <= target_width:
        return img
    scale = target_width / float(w)
    new_w = target_width
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _frames_from_video(path: str, max_frames: int = 6, resize_width: int = 512, jpeg_quality: int = 85) -> List[bytes]:
    cap = _load_video_capture(path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Fallback: probe frames by reading until exhaustion (slow path)
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
                if len(frames) >= max_frames:
                    break
        else:
            indices = _sample_frame_indices(total_frames, max_frames)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                frames.append(frame)
        # Post-process frames
        out_jpegs: List[bytes] = []
        for frame in frames:
            frame = _resize_keep_aspect(frame, resize_width)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if ok:
                out_jpegs.append(buf.tobytes())
        return out_jpegs
    finally:
        cap.release()


def _b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


def _gemini_compare(
    api_key: str,
    model: str,
    task_description: str,
    video1_frames_jpeg: List[bytes],
    video2_frames_jpeg: List[bytes],
    request_timeout_s: float = 60.0,
    temperature: float = 0.0,
    candidate_count: int = 1,
    max_output_tokens: Optional[int] = 8,
    generation_config_overrides: Optional[Dict] = None,
) -> str:
    # Build request body for Gemini generateContent
    # One user message with interleaved text and inline images for both videos.
    parts: List[Dict] = []

    system_instruction = {
        "role": "system",
        "parts": [
            {
                "text": (
                    "You are an expert robotics evaluator analyzing two videos of a robotic manipulation task. "
                    "EVALUATION CRITERIA:\n"
                    "1. **Grasp Quality**: Are the door handles grasped firmly by both hands?\n"
                    "2. **Correct Motion**: Is the door being pulled inward (toward camera)?\n"
                    "3. **Door Opening**: How far does the door open?\n"
                    "4. **Peak Performance**: Judge by the BEST moment achieved in each video\n"
                    "5. **Sustained Progress**: Brief success is still better than no success\n"
                    "\n"
                    "SUCCESS INDICATORS (in priority order):\n"
                    "- CRITICAL: Both hands successfully grasp door handles\n"
                    "- HIGH: Door moves inward (any amount)\n"
                    "- MEDIUM: Door opens 15+ degrees\n"
                    "- IDEAL: Door opens 45+ degrees and stays open\n"
                    "\n"
                    "COMMON FAILURES TO RECOGNIZE:\n"
                    "- Hands miss or just touch handles (without grasping)\n"
                    "- Pushing instead of pulling\n"
                    "- One hand grasps but other doesn't\n"
                    "- Door opens briefly then closes\n"
                    "\n"
                    "INSTRUCTIONS:\n"
                    "- Watch both videos completely\n"
                    "- Identify the peak moment of progress in each\n"
                    "- Compare peak achievements, not final states\n"
                    "- Recognize partial progress (touching < grasping < pulling < opening)\n"
                    "- If both videos show similar progress or both fail completely, choose 0\n"
                    "Return strictly one token: [1] if the FIRST video better completes the task, [2] if the SECOND video does, or [0] if they are similar/both fail. "
                    "Do not explain. Do not include any other characters."
                )
            }
        ],
    }

    parts.append({
        "text": (
            f"Task description: {task_description}\n"
            "Decide which video demonstrates better progress toward opening the door. Answer only [1], [2], or [0].\n"
            "Video 1 frames follow:"
        )
    })
    for img_bytes in video1_frames_jpeg:
        parts.append({
            "inlineData": {"mimeType": "image/jpeg", "data": _b64(img_bytes)}
        })

    parts.append({"text": "Video 2 frames follow:"})
    for img_bytes in video2_frames_jpeg:
        parts.append({
            "inlineData": {"mimeType": "image/jpeg", "data": _b64(img_bytes)}
        })

    gen_cfg = {
        "temperature": float(temperature),
        "candidateCount": int(candidate_count),
    }
    if max_output_tokens is not None:
        gen_cfg["maxOutputTokens"] = int(max_output_tokens)
    if generation_config_overrides:
        gen_cfg.update(generation_config_overrides)

    req_body = {
        "contents": [{"role": "user", "parts": parts}],
        "systemInstruction": system_instruction,
        "generationConfig": gen_cfg,
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    data = json.dumps(req_body).encode("utf-8")
    req = _urlreq.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with _urlreq.urlopen(req, timeout=request_timeout_s) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except _urlerr.HTTPError as e:
        # Surface server error body for diagnostics (e.g., unknown model/field)
        try:
            body = e.read().decode("utf-8", "ignore")
        except Exception:
            body = ""
        raise RuntimeError(f"HTTPError {e.code}: {body or getattr(e, 'reason', 'Bad Request')}")
    except _urlerr.URLError as e:
        raise RuntimeError(f"URLError: {getattr(e, 'reason', str(e))}")

    # Extract text
    candidates = out.get("candidates", [])
    if not candidates:
        print(f"DEBUG: No candidates returned. Response: {out}", flush=True)
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    final_text = "\n".join([t for t in texts if t])
    
    if not final_text:
        print(f"DEBUG: Candidates exist but no text found (likely Safety). Response: {out}", flush=True)
        
    return final_text


def _parse_choice(text: str) -> Optional[int]:
    if not text:
        return None
    # Normalize
    s = text.strip().lower()
    # Common bracketed forms
    if "[[1]]" in s or "[1]" in s or s == "1":
        return 1
    if "[[2]]" in s or "[2]" in s or s == "2":
        return 2
    if "[[0]]" in s or "[0]" in s or s == "0":
        return 0
    # Fallback: first digit occurrence
    m = re.search(r"\b([012])\b", s)
    if m:
        return int(m.group(1))
    return None


def _compare_pair(
    api_key: str,
    model: str,
    task_description: str,
    vid_a_path: str,
    vid_b_path: str,
    frame_params: Dict,
    temperature: float = 0.0,
    candidate_count: int = 1,
    max_output_tokens: Optional[int] = 8,
    generation_config_overrides: Optional[Dict] = None,
    retries: int = 2,
    retry_delay_s: float = 1.0,
) -> Tuple[Optional[int], Optional[int], str, str]:
    # Preprocess frames once per video
    a_frames = _frames_from_video(
        vid_a_path,
        max_frames=frame_params.get("max_frames", 6),
        resize_width=frame_params.get("resize_width", 512),
        jpeg_quality=frame_params.get("jpeg_quality", 85),
    )
    b_frames = _frames_from_video(
        vid_b_path,
        max_frames=frame_params.get("max_frames", 6),
        resize_width=frame_params.get("resize_width", 512),
        jpeg_quality=frame_params.get("jpeg_quality", 85),
    )

    # Normal order: (A, B)
    text_normal = ""
    for attempt in range(retries + 1):
        try:
            text_normal = _gemini_compare(
                api_key,
                model,
                task_description,
                a_frames,
                b_frames,
                temperature=temperature,
                candidate_count=candidate_count,
                max_output_tokens=max_output_tokens,
                generation_config_overrides=generation_config_overrides,
            )
            break
        except Exception as e:
            if attempt >= retries:
                text_normal = f"ERROR: {e}"
            else:
                time.sleep(retry_delay_s)

    pred_normal = _parse_choice(text_normal)

    # Skip reversed order to save time/cost since we trust normal order
    pred_reversed = None
    text_reversed = "SKIPPED"

    return pred_normal, pred_reversed, text_normal, text_reversed


def main():
    # Check for pair comparison mode (task, vid1, vid2) - called by reward_tuner.py
    if len(sys.argv) == 4 and os.path.isfile(sys.argv[2]) and os.path.isfile(sys.argv[3]):
        task_description = sys.argv[1]
        vid1_path = sys.argv[2]
        vid2_path = sys.argv[3]
        
        try:
            api_key = _read_env_api_key()
        except RuntimeError as e:
            print(e)
            # Write 5 (error) to output if API key fails
            with open("./utils/vlm_response.txt", "w") as f:
                f.write("5")
            return

        model = "gemini-robotics-er-1.5-preview" 
        
        frame_params = {"max_frames": 6, "resize_width": 512, "jpeg_quality": 85}
        
        print(f"Comparing {vid1_path} vs {vid2_path}...")
        pred_normal, pred_reversed, raw_normal, raw_reversed = _compare_pair(
            api_key, model, task_description, vid1_path, vid2_path, frame_params, temperature=0.0, max_output_tokens=8192
        )
        
        # Determine result
        # Strictly rely on the normal order.
        print(f"DEBUG: raw_normal='{raw_normal}'", flush=True)
        
        if pred_normal == 1:
            result = 1
        elif pred_normal == 2:
            result = 2
        elif pred_normal == 0:
            result = 0
        else:
            # Fallback: if model fails to output 1, 2 or 0 (returns None), default to 1 to break tie
            print(f"Warning: Model returned invalid output '{raw_normal}'. Defaulting to 1.")
            result = 1
            
        print(f"Result: {result} (Normal: {pred_normal}, Reversed: {pred_reversed})")
        
        # Output to file
        with open("./utils/vlm_response.txt", "w") as f:
            f.write(str(result))
        return

    # Manual override section: set to True to manually specify parameters here
    MANUAL_OVERRIDE = True
    TASK_DESCRIPTION = "The task is: There is a closed door in front, the two hands should grab the door handles and pull the door until it is fully open. Which one is closer to the goal, or seems like it's on the right track?"
    VIDEO_FOLDER = "/home/gx22/Desktop/isaacgym/python/Eureka/eureka/comparison_videos"
    MODEL_NAME = "gemini-robotics-er-1.5-preview"
    TEMPERATURE = 1.0
    CANDIDATE_COUNT = 1
    MAX_OUTPUT_TOKENS = 8192
    GENERATION_CONFIG_OVERRIDES = None

    # Usage: python comparison.py <task_description> <video_folder> [--max_frames 6] [--resize_width 512]
    if not MANUAL_OVERRIDE and len(sys.argv) < 3:
        print("Usage: python comparison.py <task_description> <video_folder> [--max_frames 6] [--resize_width 512]")
        sys.exit(1)

    if MANUAL_OVERRIDE:
        task_description = TASK_DESCRIPTION
        video_folder = VIDEO_FOLDER
    else:
        task_description = sys.argv[1]
        video_folder = sys.argv[2]
    max_frames = 6
    resize_width = 512
    for i in range(3, len(sys.argv)):
        if sys.argv[i] == "--max_frames" and (i + 1) < len(sys.argv):
            max_frames = int(sys.argv[i + 1])
        if sys.argv[i] == "--resize_width" and (i + 1) < len(sys.argv):
            resize_width = int(sys.argv[i + 1])

    api_key = _read_env_api_key()
    model = MODEL_NAME

    videos = _list_videos(video_folder)

    # Determine indices from filenames; fallback to position if absent
    indices: Dict[str, int] = {}
    for idx, path in enumerate(videos):
        parsed = _extract_index_from_filename(path)
        indices[path] = parsed if parsed is not None else (idx + 1)

    # Sort by extracted index to enforce the ground-truth ordering rule
    videos.sort(key=lambda p: indices[p])

    # Prepare logging
    result_path = os.path.join(video_folder, "comparision_result.txt")
    with open(result_path, "w") as f:
        f.write(f"Task: {task_description}\n")
        f.write(f"Model: {model}\n")
        f.write("Video index mapping (smaller index should be better):\n")
        for p in videos:
            f.write(f"  {indices[p]} -> {os.path.basename(p)}\n")
        f.write("\n")

    frame_params = {"max_frames": max_frames, "resize_width": resize_width, "jpeg_quality": 85}

    normal_correct = 0
    normal_total = 0
    reversed_correct = 0
    reversed_total = 0

    for a_path, b_path in combinations(videos, 2):
        a_idx = indices[a_path]
        b_idx = indices[b_path]

        # Ground truth: smaller index is better
        expected_normal = 1  # show (a, b), a has smaller index because of sort
        expected_reversed = 2  # show (b, a), second should be better

        pred_normal, pred_reversed, raw_normal, raw_reversed = _compare_pair(
            api_key,
            model,
            task_description,
            a_path,
            b_path,
            frame_params,
            temperature=TEMPERATURE,
            candidate_count=CANDIDATE_COUNT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            generation_config_overrides=GENERATION_CONFIG_OVERRIDES,
        )

        normal_is_correct = (pred_normal == expected_normal)
        reversed_is_correct = (pred_reversed == expected_reversed)

        normal_total += 1
        reversed_total += 1
        if normal_is_correct:
            normal_correct += 1
        if reversed_is_correct:
            reversed_correct += 1

        line_normal = (
            f"NORMAL: ({a_idx} vs {b_idx}) expected=[{expected_normal}] predicted=[{pred_normal}] "
            f"correct={normal_is_correct} | {os.path.basename(a_path)} vs {os.path.basename(b_path)} | raw='{(raw_normal or '').strip()}'\n"
        )
        line_reversed = (
            f"REVERSED: ({a_idx} vs {b_idx}) expected=[{expected_reversed}] predicted=[{pred_reversed}] "
            f"correct={reversed_is_correct} | {os.path.basename(b_path)} vs {os.path.basename(a_path)} | raw='{(raw_reversed or '').strip()}'\n"
        )

        print(line_normal.strip())
        print(line_reversed.strip())
        with open(result_path, "a") as f:
            f.write(line_normal)
            f.write(line_reversed)

    # Accuracy summary
    normal_acc = (normal_correct / normal_total) if normal_total else 0.0
    reversed_acc = (reversed_correct / reversed_total) if reversed_total else 0.0
    combined_correct = normal_correct + reversed_correct
    combined_total = normal_total + reversed_total
    combined_acc = (combined_correct / combined_total) if combined_total else 0.0

    summary_lines = [
        f"accuracy in normal order: {normal_acc:.4f} ({normal_correct}/{normal_total})",
        f"accuracy in reversed order: {reversed_acc:.4f} ({reversed_correct}/{reversed_total})",
        f"accuracy combined: {combined_acc:.4f} ({combined_correct}/{combined_total})",
    ]

    for s in summary_lines:
        print(s)
    with open(result_path, "a") as f:
        f.write("\n")
        for s in summary_lines:
            f.write(s + "\n")


if __name__ == "__main__":
    main()




