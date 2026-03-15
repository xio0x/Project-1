import os
import numpy as np
import matplotlib.pyplot as plt

MIN_SEGMENT_LENGTH = 16   # stop recursing if a segment is this short


# ---------------------------------------------------------------------------
# Core recursive algorithm
# ---------------------------------------------------------------------------

def recursive_segmentation(signal, start, end, threshold, segments):
    """
    Divide-and-Conquer segmentation.

    Algorithm
    ---------
    1. Compute the variance of signal[start:end].
    2. If variance > threshold  →  split at the midpoint and recurse on
       both halves (divide step).
    3. Else (or if the segment is too short to split further)  →  mark
       the current range as a *stable* segment (base case).
    """
    current = signal[start:end]

    # Base case: segment is too short to split
    if len(current) <= MIN_SEGMENT_LENGTH:
        segments.append((start, end))
        return

    variance = np.var(current)

    if variance > threshold:
        mid = (start + end) // 2
        recursive_segmentation(signal, start, mid, threshold, segments)
        recursive_segmentation(signal, mid, end, threshold, segments)
    else:
        # Variance is low enough — this segment is "stable"
        segments.append((start, end))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def summarize_segment_rul(df, segments):
    """
    For each segment determine the majority RUL category it contains.
    """
    summary = []
    for start, end in segments:
        segment_labels = df.iloc[start:end]["rul_category"].tolist()
        counts = {}
        for label in segment_labels:
            counts[label] = counts.get(label, 0) + 1
        majority_label = max(counts, key=counts.get)
        summary.append((start, end, majority_label, counts))
    return summary


def choose_10_sensors(df):
    """Select the first 10 sensor columns (sensor_00 … sensor_09)."""
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
    return sensor_cols[:10]


def save_segmentation_plot(signal, segments, sensor_name):
    os.makedirs("outputs/task1", exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.plot(signal, linewidth=0.8, color="steelblue", label="Signal")
    for start, end in segments:
        plt.axvline(x=start, color="red", linestyle="--", linewidth=0.6, alpha=0.6)
    # Mark the final boundary as well
    if segments:
        plt.axvline(x=segments[-1][1], color="red", linestyle="--",
                    linewidth=0.6, alpha=0.6)
    plt.title(f"Divide-and-Conquer Segmentation — {sensor_name}")
    plt.xlabel("Time Index")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/task1/{sensor_name}_segmentation.png", dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Task entry point
# ---------------------------------------------------------------------------

def run_segmentation_task(df):
    """
    Task 1 — Divide-and-Conquer Segmentation

    Step 1 : Select 10 sensors (sensor_00 to sensor_09).
    Step 2 : For each sensor, run the recursive segmentation algorithm.
             Threshold = 35 % of the signal's global variance — segments
             whose variance exceeds this fraction are split further.
    Step 3 : Visualise the segmentation.
    Step 4 : Compute the Segmentation Complexity Score (= number of segments).
    Step 5 : Discuss whether segments align with RUL categories.
    Step 6 : Discuss temporal dynamics in terms of complexity score.
    """
    sensors = choose_10_sensors(df)
    print("Selected 10 sensors:")
    for sensor in sensors:
        print(f"  - {sensor}")
    print()

    output_lines = ["TASK 1 - Divide-and-Conquer Segmentation\n"]

    for sensor in sensors:
        signal = df[sensor].to_numpy()

        # Threshold: 35 % of the signal's own variance.
        # This is a relative threshold so it adapts to each sensor's scale.
        threshold = np.var(signal) * 0.35

        segments = []
        recursive_segmentation(signal, 0, len(signal), threshold, segments)
        segments = sorted(segments, key=lambda x: x[0])

        complexity_score = len(segments)
        segment_rul_info = summarize_segment_rul(df, segments)

        print(f"{sensor}")
        print(f"  Segmentation Complexity Score: {complexity_score}")
        output_lines.append(sensor)
        output_lines.append(f"  Segmentation Complexity Score: {complexity_score}")

        # Print up to 5 segment previews in the console; all go to the file
        for i, (start, end, majority_label, counts) in enumerate(segment_rul_info):
            line = (f"  Segment {i + 1:3d}: [{start:5d}, {end:5d}) "
                    f"-> majority RUL category: {majority_label}")
            output_lines.append(line)
            if i < 5:
                print(line)

        if len(segment_rul_info) > 5:
            print(f"  ... ({len(segment_rul_info) - 5} more segments in summary file)")

        print()
        output_lines.append("")

        save_segmentation_plot(signal, segments, sensor)

    os.makedirs("outputs/task1", exist_ok=True)
    with open("outputs/task1/task1_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("Task 1 plots saved in outputs/task1/")
    print("Task 1 summary saved in outputs/task1/task1_summary.txt")