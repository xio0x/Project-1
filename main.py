import os
import pandas as pd
from segmentation import run_segmentation_task
from clustering import run_clustering_task
from kadane_analysis import run_kadane_task
from utils import (
    load_first_10000_rows,
    assign_rul_categories,
    print_toy_example_checks,
)


def main():
    file_name = "rul_hrs.csv"

    if not os.path.exists(file_name):
        print("Dataset file not found: rul_hrs.csv")
        print("Place the CSV in the same folder as main.py and re-run.")
        return

    # ---- Load and label data ------------------------------------------------
    df = load_first_10000_rows(file_name)
    df, q10, q50, q90 = assign_rul_categories(df)

    print("Dataset loaded — using first 10,000 rows.")
    print(f"  Q10 (Extremely Low  / Moderately Low  boundary) = {q10:.4f}")
    print(f"  Q50 (Moderately Low / Moderately High boundary) = {q50:.4f}")
    print(f"  Q90 (Moderately High/ Extremely High  boundary) = {q90:.4f}")
    print()

    # ---- Toy example verification -------------------------------------------
    print_toy_example_checks()

    # ---- Tasks --------------------------------------------------------------
    print("\n================ TASK 1: Divide-and-Conquer Segmentation ================\n")
    run_segmentation_task(df)

    print("\n================ TASK 2: Divide-and-Conquer Clustering  ================\n")
    run_clustering_task(df)

    print("\n================ TASK 3: Maximum Subarray (Kadane)      ================\n")
    run_kadane_task(df)

    print("\nProject complete.")


if __name__ == "__main__":
    main()