import os
import numpy as np


# ---------------------------------------------------------------------------
# Distance / centroid helpers
# ---------------------------------------------------------------------------

def squared_distance(a, b):
    diff = a - b
    return np.sum(diff * diff)


def compute_centroid(data):
    return np.mean(data, axis=0)


# ---------------------------------------------------------------------------
# Core divisive split
# ---------------------------------------------------------------------------

def split_cluster(indices, data):
    """
    Split one cluster (identified by a list of row indices) into two sub-clusters.

    Strategy
    --------
    1. Compute the centroid of the cluster.
    2. Find the point farthest from the centroid — that becomes seed2.
       seed1 is the centroid itself.
    3. Assign every point to whichever seed it is closer to.
    4. Guard against degenerate splits (one empty half) by falling back to
       a simple index-based halving.

    Returns two lists of original row indices.
    """
    cluster_data = data[indices]
    centroid = compute_centroid(cluster_data)

    # Find the farthest point from the centroid
    distances = np.array([squared_distance(cluster_data[i], centroid)
                          for i in range(len(cluster_data))])
    farthest_pos = int(np.argmax(distances))
    seed2 = cluster_data[farthest_pos]

    group1_idx = []
    group2_idx = []
    for pos, orig_idx in enumerate(indices):
        d1 = squared_distance(cluster_data[pos], centroid)
        d2 = squared_distance(cluster_data[pos], seed2)
        if d1 <= d2:
            group1_idx.append(orig_idx)
        else:
            group2_idx.append(orig_idx)

    # Guard: if one group is empty fall back to index-split
    if len(group1_idx) == 0 or len(group2_idx) == 0:
        half = len(indices) // 2
        group1_idx = list(indices[:half])
        group2_idx = list(indices[half:])

    return group1_idx, group2_idx


# ---------------------------------------------------------------------------
# Recursive top-down (divisive) clustering
# ---------------------------------------------------------------------------

def recursive_divisive_clustering(data, target_clusters):
    """
    Recursively split the dataset into `target_clusters` groups.

    At each step the *largest* current cluster is split.

    Parameters
    ----------
    data            : np.ndarray, shape (N, F)
    target_clusters : int

    Returns
    -------
    List of lists, where each inner list contains the original row indices
    belonging to that cluster.
    """
    all_indices = list(range(len(data)))
    clusters = [all_indices]          # start with one cluster containing all rows

    while len(clusters) < target_clusters:
        # Pick the largest cluster to split
        largest_idx = int(np.argmax([len(c) for c in clusters]))
        cluster_to_split = clusters.pop(largest_idx)

        left, right = split_cluster(cluster_to_split, data)
        clusters.append(left)
        clusters.append(right)

    return clusters


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def majority_label(labels):
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    best_label = max(counts, key=counts.get)
    return best_label, counts


# ---------------------------------------------------------------------------
# Task entry point
# ---------------------------------------------------------------------------

def run_clustering_task(df):
    """
    Task 2 — Divide-and-Conquer Clustering of Segments

    Step 1 : Apply the divisive clustering algorithm to all 10 000 time
             instances, each described by its 52 sensor measurements.
             Produce 4 clusters.
    Step 2 : For each cluster compute the majority true class based on
             the four RUL categories.
    Step 3 : Discuss how each cluster maps to a class.
    """
    os.makedirs("outputs/task2", exist_ok=True)

    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
    data = df[sensor_cols].to_numpy()
    label_array = df["rul_category"].to_numpy()

    clusters = recursive_divisive_clustering(data, 4)

    output_lines = ["TASK 2 - Divide-and-Conquer Clustering of Segments\n"]
    print("Created 4 clusters\n")

    for i, row_indices in enumerate(clusters):
        cluster_labels = [label_array[idx] for idx in row_indices]
        dominant_label, counts = majority_label(cluster_labels)

        print(f"Cluster {i + 1}")
        print(f"  Size              : {len(row_indices)}")
        print(f"  Majority RUL class: {dominant_label}")
        print(f"  Class counts      : {counts}")
        print()

        output_lines.append(f"Cluster {i + 1}")
        output_lines.append(f"  Size: {len(row_indices)}")
        output_lines.append(f"  Majority true class: {dominant_label}")
        output_lines.append(f"  Class counts: {counts}")
        output_lines.append("")

    with open("outputs/task2/task2_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("Task 2 summary saved in outputs/task2/task2_summary.txt")