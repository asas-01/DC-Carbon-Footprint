"""
Author: asas
Description: Data Center Carbon Footprint Analysis

run_all.py - Run the complete analysis pipeline in sequence.

Usage: python src/run_all.py
"""

import subprocess
import sys
import os

SCRIPTS = [
    "01_prepare_data.py",
    "02_analysis.py",
    "03_projections.py",
    "04_generate_figures.py",
]

def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    for script in SCRIPTS:
        path = os.path.join(src_dir, script)
        print(f"\n{'#' * 60}")
        print(f"# Running {script}")
        print(f"{'#' * 60}\n")
        result = subprocess.run([sys.executable, path], cwd=src_dir)
        if result.returncode != 0:
            print(f"\nERROR: {script} failed with exit code {result.returncode}")
            sys.exit(1)

    print("\n" + "#" * 60)
    print("# Pipeline complete!")
    print("#" * 60)


if __name__ == "__main__":
    main()
