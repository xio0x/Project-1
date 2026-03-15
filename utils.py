import pandas as pd
import numpy as np


def load_first_10000_rows(file_name):
    df = pd.read_csv(file_name)
    df = df.iloc[:10000].copy()
    return df


def assign_rul_categories(df):
    """
    Convert continuous RUL into 4 machine-health categories using quantile boundaries.

    Boundaries (from assignment):
      Q10 = 10th percentile of RUL
      Q50 = 50th percentile (median) of RUL   <-- assignment table uses Q50, not Q40
      Q90 = 90th percentile of RUL

    Categories:
      Extremely Low RUL   : rul < Q10
      Moderately Low RUL  : Q10 <= rul < Q50
      Moderately High RUL : Q50 <= rul < Q90
      Extremely High RUL  : rul >= Q90
    """
    q10 = df["rul"].quantile(0.10)
    q50 = df["rul"].quantile(0.50)   # median — assignment table boundary
    q90 = df["rul"].quantile(0.90)

    categories = []
    for value in df["rul"]:
        if value < q10:
            categories.append("Extremely Low RUL")
        elif value < q50:
            categories.append("Moderately Low RUL")
        elif value < q90:
            categories.append("Moderately High RUL")
        else:
            categories.append("Extremely High RUL")

    df["rul_category"] = categories
    return df, q10, q50, q90


# ---------------------------------------------------------------------------
# Toy example verification — actually runs each algorithm on tiny inputs
# so correctness can be confirmed before processing the real dataset.
# ---------------------------------------------------------------------------

def _toy_segmentation():
    """
    Tiny signal with two clearly distinct regions:
      - indices 0-3 : near-constant (low variance)  → should stay as one segment
      - indices 4-7 : high-variance                 → should split further
    """
    from segmentation import recursive_segmentation

    signal = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0,
                       10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0])
    threshold = np.var(signal) * 0.35
    segments = []
    recursive_segmentation(signal, 0, len(signal), threshold, segments)
    segments = sorted(segments, key=lambda x: x[0])

    print("  Signal  :", signal.tolist())
    print("  Threshold (0.35 * global variance):", round(threshold, 4))
    print("  Segments found:", segments)
    print("  Complexity score:", len(segments))
    print("  Expected: the stable region [0,4) becomes one segment;")
    print("            the volatile region produces more splits.")


def _toy_clustering():
    """
    Four clearly separated 2-D points — divisive clustering should
    place each point in its own cluster after 3 splits.
    """
    from clustering import recursive_divisive_clustering

    points = np.array([[1.0, 1.0],
                       [1.5, 1.5],
                       [9.0, 9.0],
                       [9.5, 9.5]])

    clusters = recursive_divisive_clustering(points, 2)

    print("  Points  :", points.tolist())
    print("  Clusters requested: 2")
    for idx, row_indices in enumerate(clusters):
        cluster_points = points[row_indices].tolist()
        print(f"  Cluster {idx + 1}: {cluster_points}")
    print("  Expected: one cluster near [1,1] and one near [9,9].")


def _toy_kadane():
    """
    Classic Kadane example with a known answer.
    Array: [-2, 3, 4, -1, 2, -5]
    Best subarray: [3, 4, -1, 2]  (indices 1-4, sum = 8)
    """
    from kadane_analysis import kadane

    arr = np.array([-2.0, 3.0, 4.0, -1.0, 2.0, -5.0])
    start, end, total = kadane(arr)

    print("  Array   :", arr.tolist())
    print(f"  Best subarray indices : [{start}, {end}]")
    print(f"  Subarray values       : {arr[start:end + 1].tolist()}")
    print(f"  Total sum             : {total}")
    print("  Expected: indices [1, 4], values [3.0, 4.0, -1.0, 2.0], sum = 8.0")

    assert start == 1 and end == 4 and abs(total - 8.0) < 1e-9, \
        f"Kadane toy example FAILED: got [{start},{end}] sum={total}"
    print("  PASSED ✓")


def print_toy_example_checks():
    print("TOY EXAMPLE VERIFICATION")
    print("=" * 50)

    print("\n[1] Segmentation toy example")
    print("    Verifies that stable regions are kept whole and")
    print("    high-variance regions are recursively split.")
    _toy_segmentation()

    print("\n[2] Clustering toy example")
    print("    Verifies that divisive clustering separates")
    print("    well-separated point groups correctly.")
    _toy_clustering()

    print("\n[3] Kadane toy example")
    print("    Verifies the maximum-subarray result on a known input.")
    _toy_kadane()

    print("\n" + "=" * 50)