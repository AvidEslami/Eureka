# How to Make VLM Smarter - Implementation Guide

## 1. Enable Extended Reasoning (Thinking Mode)

Replace line 169 in vlm.py:
```python
# OLD:
model="gemini-2.5-pro"

# NEW (with thinking):
model="gemini-2.5-pro-thinking"  # Or use thinking mode parameter
```

## 2. Improve the Prompt with Chain-of-Thought

**Current Prompt (line 454):**
```python
test_prompt = "Evaluate the two trajectories demonstrated in the videos and decide which one is closer to the goal..."
```

**Improved Prompt with Reasoning:**
```python
test_prompt = """You are an expert robotics evaluator. Analyze the two videos showing robotic manipulation tasks.

TASK GOAL: {task_description}

EVALUATION CRITERIA:
1. **Progress towards goal**: How close does each trajectory get to completing the task?
2. **Key milestones**: 
   - Are door handles grasped firmly?
   - Is the door being pulled in the correct direction (inward)?
   - How far does the door open?
3. **Peak performance**: Judge by the BEST moment in each video, not the final state
4. **Partial credit**: Recognize partial progress (e.g., touching handle vs grasping vs pulling)

INSTRUCTIONS:
- Watch both videos completely
- Identify the best moment in each video
- Compare peak progress, not final states
- Consider smoothness and control
- If videos are very similar or both fail, respond with 0

OUTPUT FORMAT:
Respond with ONLY the number: 1 (if Video 1 is better), 2 (if Video 2 is better), or 0 (if similar/both fail)

Video 1: {video1_name}
Video 2: {video2_name}

Which video demonstrates better progress toward the goal?"""
```

## 3. Add More Context to Task Descriptions

**Current (line 30):**
```python
return "Open the door using the two robotic hands, the door handles must first be grabbed, then pulled inwards in order to be opened."
```

**Improved:**
```python
return """TASK: Open an inward-opening door using two robotic hands.

SUCCESS CRITERIA (in order of importance):
1. CRITICAL: Both hands grasp the door handles firmly
2. HIGH: Door is pulled toward the camera (inward motion)
3. MEDIUM: Door opens at least 30 degrees
4. IDEAL: Door opens fully (90+ degrees) and stays open

COMMON FAILURE MODES:
- Hands miss the handles
- Hands touch but don't grasp
- Pulling in wrong direction (pushing instead)
- Door opens briefly then closes

EVALUATION NOTES:
- Grasping handles = significant progress even if door doesn't open
- Small door opening (10-20°) is better than no opening
- Sustained grasp is better than brief touch"""
```

## 4. Adjust Temperature for Better Consistency

**Current:** temperature=0.0 (fully deterministic)

**Options:**
```python
# For maximum consistency (current):
"temperature": 0.0

# For slightly more nuanced evaluation:
"temperature": 0.1  # Slight variation, still consistent

# For more creative reasoning:
"temperature": 0.3  # More variation in evaluation
```

## 5. Increase Output Tokens for Reasoning

**Current:** max_output_tokens=1024

**Better:**
```python
"max_output_tokens": 2048  # Allow more detailed reasoning if needed
```

## 6. Use Video Caching (Reduces Cost & Latency)

Add caching for repeated video comparisons:
```python
# In query_vlm_with_video, after uploading:
vf = client.files.upload(file=path)
# Add this:
# Enable caching for this file (valid for ~1 hour)
# This reduces cost if comparing same videos multiple times
```

## 7. Add Video Preprocessing

Before sending videos, you could:
```python
# Clip to key moments (saves tokens/time)
# Extract key frames
# Adjust playback speed
# Compress if too large
```

## 8. Request Reasoning Explanation (then parse)

Change schema to get reasoning:
```python
schema = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of which video is better and why"
        },
        "winner": {
            "type": "string",
            "enum": ["1", "2", "0"],
            "description": "1 = video 1 is better, 2 = video 2 is better, 0 = similar"
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Confidence in the judgment"
        }
    },
    "required": ["reasoning", "winner", "confidence"]
}
```

Then parse:
```python
result = json.loads(part.text)
print(f"Reasoning: {result['reasoning']}")
print(f"Confidence: {result['confidence']}")
winner = int(result["winner"])
```

## 9. Multi-Query Ensemble (Already Partially Done!)

Your code already queries twice with flipped order. You could:
- Query 3-5 times
- Take majority vote
- Weight by confidence

## 10. Add Visual Markers to Videos

If you control video generation:
- Add overlays showing key metrics (door angle, hand position)
- Highlight successful moments
- Add timestamp markers

## Quick Wins (Easiest to Implement)

1. ✅ **Better prompt** (lines 453-454) - Copy improved prompt above
2. ✅ **Increase temperature** to 0.1-0.2 (line 173)
3. ✅ **Add reasoning to schema** (line 106-116)
4. ✅ **Better task descriptions** (line 30)

## Implementation Priority

**High Impact, Low Effort:**
1. Improved prompt with evaluation criteria ⭐⭐⭐
2. Better task descriptions ⭐⭐⭐
3. Add reasoning output ⭐⭐

**Medium Impact, Medium Effort:**
4. Adjust temperature slightly ⭐⭐
5. Video preprocessing ⭐⭐

**High Impact, High Effort:**
6. Enable thinking mode ⭐⭐⭐
7. Implement caching ⭐

