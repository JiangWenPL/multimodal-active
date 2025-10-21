import numpy as np
from itertools import product

def compute_pairwise_distances_vectorized(point_indices, point_arrays):
    """
    Compute minimum pairwise distances between selected points using vectorized operations
    
    Args:
        point_indices: List of indices, one for each label's selected point
        point_arrays: List of np.arrays containing points for each label
    
    Returns:
        float: Minimum distance between any pair of selected points
    """
    selected_points = np.array([point_arrays[i][idx] for i, idx in enumerate(point_indices)])
    
    # Compute all pairwise differences
    diff = selected_points[:, np.newaxis] - selected_points
    
    # Compute Euclidean distances
    distances = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(distances, np.inf)
    
    return np.min(distances)

def compute_all_distances_batch(point_arrays):
    """
    Precompute all pairwise distances between points of different labels in a vectorized way
    
    Args:
        point_arrays: List of np.arrays containing points for each label
    
    Returns:
        list: List of distance matrices between each pair of label groups
    """
    n_labels = len(point_arrays)
    all_distances = []
    
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            # Compute distances between all points in labels i and j
            diff = point_arrays[i][:, np.newaxis] - point_arrays[j]
            distances = np.sqrt(np.sum(diff**2, axis=2))
            all_distances.append(distances)
    
    return all_distances

def select_maximin_points_vectorized(point_arrays):
    """
    Select one point from each labeled group such that the minimum distance
    between any pair of selected points is maximized, using vectorized operations.
    
    Args:
        point_arrays: List of np.arrays, where each array contains 2D points
                     of one label with shape (n_points, 2)
    
    Returns:
        list: Indices of selected points for each label
        list: Selected points
        float: Maximum minimum distance achieved
    """
    # Get number of points in each array
    n_points = [arr.shape[0] for arr in point_arrays]
    
    # Precompute all pairwise distances between different label groups
    distance_matrices = compute_all_distances_batch(point_arrays)
    
    # Generate all possible combinations of point indices
    all_combinations = list(product(*[range(n) for n in n_points]))
    all_combinations = np.array(all_combinations)
    
    # Process combinations in batches to avoid memory overflow
    batch_size = 10000
    n_combinations = len(all_combinations)
    max_min_dist = float('-inf')
    best_indices = None
    
    for start_idx in range(0, n_combinations, batch_size):
        end_idx = min(start_idx + batch_size, n_combinations)
        batch_combinations = all_combinations[start_idx:end_idx]
        
        # Initialize minimum distances for this batch
        batch_min_distances = np.full(len(batch_combinations), np.inf)
        
        # Process each pair of labels
        matrix_idx = 0
        for i in range(len(point_arrays)):
            for j in range(i + 1, len(point_arrays)):
                # Get relevant indices for this pair of labels
                indices_i = batch_combinations[:, i]
                indices_j = batch_combinations[:, j]
                
                # Get distances for these combinations
                distances = distance_matrices[matrix_idx][indices_i, indices_j]
                
                # Update minimum distances
                batch_min_distances = np.minimum(batch_min_distances, distances)
                matrix_idx += 1
        
        # Find the best combination in this batch
        batch_best_idx = np.argmax(batch_min_distances)
        batch_max_min_dist = batch_min_distances[batch_best_idx]
        
        # Update global best if necessary
        if batch_max_min_dist > max_min_dist:
            max_min_dist = batch_max_min_dist
            best_indices = batch_combinations[batch_best_idx]
    
    # Get the selected points
    selected_points = [point_arrays[i][idx] for i, idx in enumerate(best_indices)]
    
    return best_indices.tolist(), selected_points, max_min_dist

# Example usage with visualization
def example_usage_with_viz():
    import matplotlib.pyplot as plt
    
    # Create sample data: 3 labels, each with several 2D points
    np.random.seed(42)
    point_arrays = [
        np.random.rand(4, 2) * 2,
        np.random.rand(3, 2) * 2 + 2,
        np.random.rand(3, 2) * 2 + 4
    ]
    
    # Find optimal points
    indices, points, max_min_dist = select_maximin_points_vectorized(point_arrays)
    
    # Plot results
    colors = ['r', 'g', 'b']
    plt.figure(figsize=(10, 8))
    
    # Plot all points
    for i, points_array in enumerate(point_arrays):
        plt.scatter(points_array[:, 0], points_array[:, 1], 
                   c=colors[i], alpha=0.3, label=f'Label {i+1}')
    
    # Plot selected points
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1], c='black', s=100, marker='*',
               label='Selected Points')
    
    plt.title(f'Selected Points (Max Min Distance: {max_min_dist:.3f})')
    plt.legend()
    plt.grid(True)
    
    print(f"Selected point indices: {indices}")
    print(f"Selected points:\n{points}")
    print(f"Maximum minimum distance: {max_min_dist:.3f}")
    plt.show()
    
    return point_arrays, points

def min_dist_center_approximate(point_arrays):
    """
    select the points that are closest to the center of each group
    """
    centers = [np.mean(points, axis=0) for points in point_arrays]
    closest_points = []
    closest_indices = []

    for i, points in enumerate(point_arrays):
        distances = np.linalg.norm(points - centers[i], axis=1)
        min_index = np.argmin(distances)
        closest_points.append(points[min_index])
        closest_indices.append(min_index)

    return closest_indices, np.array(closest_points)

if __name__ == "__main__":
    example_usage_with_viz()