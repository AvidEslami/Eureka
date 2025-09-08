import os
import time
import subprocess
from pathlib import Path


def main():
    # Fixed settings per request
    seed = 2
    task = "Ant"
    suffix = "GPT"
    max_iterations = 1000

    # Absolute paths (user prefers absolute paths)
    root_dir = "/home/gx22/Desktop/isaacgym/python/Eureka"
    sanitized_dir = f"{root_dir}/eureka/ant_data_body_sanitized"
    train_py = f"{root_dir}/isaacgymenvs/isaacgymenvs/train.py"

    # Iterate ep005..ep500 (every 5 epochs) and train RL for each with 6-way parallelism
    concurrency = 6
    active = []  # list of dicts: {proc, file, ep}

    def reap_finished(block: bool = False):
        while True:
            removed_any = False
            for i in range(len(active) - 1, -1, -1):
                p = active[i]["proc"]
                if p.poll() is not None:
                    try:
                        active[i]["file"].close()
                    finally:
                        active.pop(i)
                        removed_any = True
            if removed_any:
                return
            if not block:
                return
            time.sleep(0.5)

    try:
        for ep in range(5, 501, 5):
            ckpt = Path(sanitized_dir) / f"Ant_nn_train_ep{ep:03d}.ptt"
            if not ckpt.exists():
                continue

            # Backpressure: keep at most `concurrency` processes
            while len(active) >= concurrency:
                reap_finished(block=True)

            # Per-process environment (do not mutate global os.environ)
            env = os.environ.copy()
            env["EUREKA_REWARD_MODEL"] = str(ckpt)

            # Log file matching prior format
            log_path = f"{root_dir}/eureka/seed_{seed}_sanitized_epoch_{ep}.txt"

            try:
                f = open(log_path, "w")
                proc = subprocess.Popen([
                    "python", "-u", train_py,
                    "hydra/output=subprocess",
                    f"task={task}{suffix}",
                    f"headless=True", "capture_video=False", "force_render=False",
                    f"seed={seed}", f"max_iterations={max_iterations}"
                ], stdout=f, stderr=f, env=env)
                active.append({"proc": proc, "file": f, "ep": ep})
            except Exception as e:
                with open(log_path, "a") as ef:
                    ef.write(f"\nERROR during training launch for epoch {ep:03d}: {e}\n")

        # Drain remaining
        while active:
            reap_finished(block=True)
    except KeyboardInterrupt:
        # Gracefully terminate all
        for item in active:
            try:
                item["proc"].terminate()
            except Exception:
                pass
        for item in active:
            try:
                item["proc"].wait(timeout=5)
            except Exception:
                try:
                    item["proc"].kill()
                except Exception:
                    pass
            finally:
                try:
                    item["file"].close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()


