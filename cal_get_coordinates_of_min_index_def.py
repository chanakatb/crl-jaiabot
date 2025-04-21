import numpy as np

# def get_coordinates_of_min_index(X, index_for_relative_positions, min_index):
#     # Get the two column indices from the row specified by min_index
#     col1, col2 = index_for_relative_positions[min_index]
    
#     # Extract coordinates from X using these indices
#     coordinates = np.array([
#         [X[0][col1], X[0][col2]],  # x coordinates
#         [X[1][col1], X[1][col2]]   # y coordinates
#     ])

#     minimum_distance = np.sqrt((X[0][col1] - X[0][col2])**2 + (X[1][col1] - X[1][col2])**2)
    
#     return coordinates, minimum_distance

import numpy as np

def get_coordinates_of_min_index(position, index_for_relative_positions):
    # Calculate relative positions and distances
    distances = []
    
    for i, idx_pair in enumerate(index_for_relative_positions):
        idx1, idx2 = idx_pair
        x1, y1 = position[0, idx1], position[1, idx1]
        x2, y2 = position[0, idx2], position[1, idx2]
        
        # Calculate relative position vector
        rel_x = x2 - x1
        rel_y = y2 - y1
        
        # Calculate distance
        distance = np.sqrt(rel_x**2 + rel_y**2)
        distances.append(distance)
        
        # Create coordinates array for this pair
        coordinates = np.array([
            [x1, x2],  # x coordinates
            [y1, y2]   # y coordinates
        ])
        
        # # Print information about each pair
        # print(f"Pair {i+1}: {idx_pair}")
        # print(f"Distance {i+1}: {distance:.4f}")
        # print(f"coordinates = np.array([")
        # print(f"    [{coordinates[0, 0]:.4f}, {coordinates[0, 1]:.4f}],  # x coordinates")
        # print(f"    [{coordinates[1, 0]:.4f}, {coordinates[1, 1]:.4f}]   # y coordinates")
        # print(f"])")
        # print()

    # Find minimum distance and its index
    min_distance = min(distances)
    min_index = distances.index(min_distance)

    # Get the corresponding pair
    min_pair = index_for_relative_positions[min_index]
    idx1, idx2 = min_pair

    # Extract coordinates for the minimum distance pair
    min_coordinates = np.array([
        [position[0, idx1], position[0, idx2]],  # x coordinates
        [position[1, idx1], position[1, idx2]]   # y coordinates
    ])
    return min_coordinates, min_index, min_distance

def main():
    # Position data
    position = np.array([
        [-85.71, -92.923, -95.8702, -110.9856, -100, -103.17611852, -112.92009693, -118.26798073, -123.691235],
        [-55.2729, -64.1704, -67.2402, -78.9206, -85, -43.18970514, -50.66845102, -55.91590601, -62.37437918]
    ])

    # Relative position indices
    index_for_relative_positions = np.array([
        [0, 1], [1, 2], [3, 2], [1, 3], [2, 0], [0, 3], 
        [0, 4], [1, 4], [2, 4], [3, 4], [0, 5], [1, 6], [2, 7], [3, 8]
    ])
    min_coordinates, min_index, min_distance = get_coordinates_of_min_index(position, index_for_relative_positions)

    print(f"Minimum distance: {min_distance:.4f}")
    print(f"Index in the relative position array: {min_index}")
    print(f"Pair indices: {index_for_relative_positions[min_index]}")
    print("\nCoordinates of the minimum distance pair:")
    print(f"coordinates = np.array([")
    print(f"    [{min_coordinates[0, 0]:.4f}, {min_coordinates[0, 1]:.4f}],  # x coordinates")
    print(f"    [{min_coordinates[1, 0]:.4f}, {min_coordinates[1, 1]:.4f}]   # y coordinates")
    print(f"])")

if __name__ == "__main__":
    main()