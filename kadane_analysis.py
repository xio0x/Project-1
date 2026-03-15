import os
import numpy as np


def kadane(arr):
    """
    Kadane's algorithm — finds the contiguous subarray with the maximum sum.

    Returns
    -------
    best_start : int   index of the first element of the max-sum subarray
    best_end   : int   index of the last  element of the max-sum subarray (inclusive)
    max_sum    : float the maximum subarray sum
    """
    if len(arr) == 0:
        raise ValueError("Input array must not be empty.")

    max_sum = arr[0]
    current_sum = arr[0]
    best_start = 0
    best_end = 0
    temp_start = 0

    for i in range(1, len(arr)):
        # Start a fresh subarray at i if arr[i] alone beats extending
        if arr[i] > current_sum + arr[i]:   # equivalent to: current_sum < 0
            current_sum = arr[i]
            temp_start = i
        else:
            current_sum = current_sum + arr[i]

        if current_sum > max_sum:
            max_sum = current_sum
            best_start = temp_start
            best_end = i

    return best_start, best_end, max_sum


def majority_label(labels):
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    best_label = max(counts, key=counts.get)
    return best_label, counts


def run_kadane_task(df):
    """
    Task 3 — Maximum Subarray (Kadane)

    For each of the 52 sensor channels:
      Step 1 : compute absolute first-difference  d[i] = |sensor[i] - sensor[i-1]|
      Step 2 : mean-centre the differences        x[i] = d[i] - mean(d)
      Step 3 : run Kadane on x to find the interval with the most intense
               sustained deviation (max-sum subarray)
      Step 4 : look up the RUL category for every time index in that interval
      Step 5 : report the dominant RUL category for the interval

    After processing all sensors, identify which sensors have their
    max-deviation interval dominated by low-RUL categories — these are
    candidate early-warning indicators.
    """
    os.makedirs("outputs/task3", exist_ok=True)

    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]

    output_lines = ["TASK 3 - Maximum Subarray (Kadane)\n"]
    low_rul_sensors = []

    for sensor in sensor_cols:
        signal = df[sensor].to_numpy()

        # Step 1 — absolute first-difference (length N-1)
        absolute_diff = np.abs(np.diff(signal))

        # Step 2 — subtract the mean so negative values mean "below average change"
        adjusted = absolute_diff - np.mean(absolute_diff)

        # Step 3 — Kadane on the adjusted difference signal
        start, end, max_sum = kadane(adjusted)

        # Step 4 — map diff indices back to original time indices.
        # diff index i corresponds to the transition between original[i] and original[i+1].
        # We cover the original rows from start to end+1 (inclusive on both original endpoints).
        rul_labels = df.iloc[start: end + 2]["rul_category"].tolist()

        # Step 5 — dominant RUL category inside the interval
        dominant_label, counts = majority_label(rul_labels)

        print(f"{sensor}")
        print(f"  Max-deviation interval (diff indices): [{start}, {end}]")
        print(f"  Corresponding original time indices  : [{start}, {end + 1}]")
        print(f"  Total deviation score: {max_sum:.4f}")
        print(f"  Majority RUL category in interval   : {dominant_label}")
        print(f"  Class counts: {counts}")
        print()

        output_lines.append(sensor)
        output_lines.append(f"  Max-deviation interval: [{start}, {end}]")
        output_lines.append(f"  Total deviation: {max_sum:.4f}")
        output_lines.append(f"  Majority RUL category in interval: {dominant_label}")
        output_lines.append(f"  Class counts: {counts}")
        output_lines.append("")

        if dominant_label in ["Extremely Low RUL", "Moderately Low RUL"]:
            low_rul_sensors.append(sensor)

    # Step 5 (global) — which sensors flag low RUL in their high-deviation window?
    output_lines.append("Sensors whose max-deviation interval is dominated by low RUL")
    output_lines.append("(potential early indicators of machine degradation):")
    if low_rul_sensors:
        for sensor in low_rul_sensors:
            output_lines.append(f"  {sensor}")
    else:
        output_lines.append("  None found — high-deviation intervals are not concentrated in low-RUL periods.")

    print("Sensors that may be early indicators of low RUL:")
    if low_rul_sensors:
        for s in low_rul_sensors:
            print(f"  {s}")
    else:
        print("  None — no sensor's max-deviation interval was dominated by low RUL.")

    with open("outputs/task3/task3_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("\nTask 3 summary saved in outputs/task3/task3_summary.txt")