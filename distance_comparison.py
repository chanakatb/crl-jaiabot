import numpy as np

B_matrix = np.array([[ 1, -1,  0,  0,  0,  0,  0,  0,  0],
              [ 0,  1, -1,  0,  0,  0,  0,  0,  0],
              [ 0,  0, -1,  1,  0,  0,  0,  0,  0],
              [ 0,  1,  0, -1,  0,  0,  0,  0,  0],
              [-1,  0,  1,  0,  0,  0,  0,  0,  0],
              [ 1,  0,  0, -1,  0,  0,  0,  0,  0],
              [ 1,  0,  0,  0, -1,  0,  0,  0,  0],
              [ 0,  1,  0,  0, -1,  0,  0,  0,  0],
              [ 0,  0,  1,  0, -1,  0,  0,  0,  0],
              [ 0,  0,  0,  1, -1,  0,  0,  0,  0],
              [ 1,  0,  0,  0,  0, -1,  0,  0,  0],
              [ 0,  1,  0,  0,  0,  0, -1,  0,  0],
              [ 0,  0,  1,  0,  0,  0,  0, -1,  0],
              [ 0,  0,  0,  1,  0,  0,  0,  0, -1]])
num_points = B_matrix.shape[1]  # Number of columns in B_matrix
# Convert args into x and y coordinates

position = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]


x = position[::2]  # even indices are x coordinates
y = position[1::2]  # odd indices are y coordinates
distances = []
all_distances = {}

for i in range(B_matrix.shape[0]):
        print(f"\ni: [{i}]")
        indices = np.nonzero(B_matrix[i])[0]
        print("Indices: ", indices)
        
        idx1, idx2 = None, None
        for j in indices:
            if B_matrix[i][j] == 1:
                idx1 = j
            elif B_matrix[i][j] == -1:
                idx2 = j
        print(f"idx1: [{idx1}]     idx2: [{idx2}]")
        if idx1 is not None and idx2 is not None:
            # Check if indices are within bounds
            if idx1 >= num_points or idx2 >= num_points:
                print(f"Warning: Index out of range for row {i}. idx1={idx1}, idx2={idx2}, num_points={num_points}")
                continue
                            
            # Calculate actual distance
            print(f"x[{idx1}]: {x[idx1]}; x[{idx2}]: {x[idx2]}")
            print(f"y[{idx1}]: {y[idx1]}; y[{idx2}]: {y[idx2]}")

            d_val = np.sqrt(((x[idx1] - x[idx2]))**2 + 
                           ((y[idx1] - y[idx2]))**2)
            print(f"d_val: [{d_val}]")
            distances.append(d_val)
            all_distances[f'd{i}'] = d_val
            # symbolic_distances[i] = d_symbolic

min_idx = np.argmin(distances)
min_dist = distances[min_idx]
print(f"\n\nmin_dist = distance[{min_idx}] = {distances[min_idx]}")

print("---"*20)


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


# Position data
position = np.array([
    [position[0], position[2], position[4], position[6], position[8], position[10], position[12], position[14], position[16]],
    [position[1], position[3], position[5], position[7], position[9], position[11], position[13], position[15], position[17]]
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
