from pathlib import Path
import subprocess

in_dir  = Path("/home/gx22/Desktop/isaacgym/python/Eureka/eureka/door_inward_videos")
out_dir = Path("/home/gx22/Desktop/isaacgym/python/Eureka/eureka/door_inward_videos_cliped")
out_dir.mkdir(exist_ok=True)

for src in sorted(in_dir.glob("*.mp4")):
    dur = float(subprocess.check_output([
        "ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1", str(src)
    ]).strip())
    t = max(0.1, dur - 2)
    dst = out_dir / src.name
    subprocess.run([
        "ffmpeg","-y","-i",str(src),"-to",f"{t:.3f}",
        "-c","copy","-movflags","+faststart",str(dst)
    ], check=True)
print("Done.")