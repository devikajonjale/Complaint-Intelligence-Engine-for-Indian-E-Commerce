"""Run full complaint intelligence pipeline sequentially."""

import subprocess
import sys

SCRIPTS = [
    "01_ingest.py",
    "02_preprocess.py",
    "03_embed.py",
    "04_reduce.py",
    "05_cluster.py",
    "06_anomaly.py",
    "07_visualize.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===")
        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode != 0:
            raise SystemExit(f"Stopped at {script} with code {result.returncode}")
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
