import numpy as np
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt



def gps_to_local_coordinates(robots_lat_lon, latitude_hub, longitude_hub, global_x_offset=0):
    """
    Transform GPS coordinates to local coordinates with configurable frame rotation. The local coordinates frame is right-hand oriented.
    
    Args:
        robots_lat_lon: numpy array of shape (2,n) containing [latitudes; longitudes]
        latitude_hub: float, hub's latitude
        longitude_hub: float, hub's longitude
        global_x_offset: float, rotation angle in degrees (0 = North, 90 = East)
        
    Returns:
        local_coords: numpy array of shape (2,n) containing [x; y] in meters
        fig: matplotlib figure showing the transformation
    """
    # Create CRS objects for WGS84 and UTM
    wgs84 = CRS.from_epsg(4326)  # WGS84 geographic coordinate system
    utm = CRS.from_epsg(32618)   # UTM zone 18N (based on your coordinates)
    
    # Create transformer object
    transformer = Transformer.from_crs(wgs84, utm, always_xy=True)
    
    # Transform hub coordinates to UTM
    hub_x, hub_y = transformer.transform(longitude_hub, latitude_hub)
    
    # Transform robot coordinates to UTM
    robot_x, robot_y = transformer.transform(robots_lat_lon[1], robots_lat_lon[0])
    
    # Calculate relative positions (this doesn't change with frame rotation)
    local_coords = np.vstack((robot_x - hub_x, robot_y - hub_y))
    
    # Convert to radians
    theta = np.radians(global_x_offset)  

    
    # Calculate coordinates in rotated frame
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    rotated_coords = R @ local_coords
    
    return rotated_coords

# Example usage
if __name__ == "__main__":
    robots_lat_lon = np.array([
        [38.787601, 38.78734, 38.787445, 38.787855],
        [-75.161919, -75.161686, -75.162581, -75.162276]
    ])
    latitude_hub = 38.78753
    longitude_hub = -75.16214
    
    # Try different rotations
    rotations = [0, 90, 180, 270]  # North and East alignment
    
    for rotation in rotations:
        local_coords = gps_to_local_coordinates(
            robots_lat_lon, latitude_hub, longitude_hub, global_x_offset=rotation
        )
        plt.show()
        print(f"\nCoordinates in rotated frame ({rotation}°) from East in counter Clockwise direction:")
        print(f"X coordinates (meters): {local_coords[0]}")
        print(f"Y coordinates (meters): {local_coords[1]}")

        print(local_coords)