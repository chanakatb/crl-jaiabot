import numpy as np
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt
from cal_gps_to_local_coordinates_def import gps_to_local_coordinates

def local_to_gps_coordinates(robots_position, latitude_hub, longitude_hub, global_x_offset=0):
    """
    Transform local coordinates back to GPS coordinates (inverse of gps_to_local_coordinates).
    The local coordinates frame is right-hand oriented.
    
    Args:
        robots_position: numpy array of shape (2,n) containing [x; y] in meters
        latitude_hub: float, hub's latitude
        longitude_hub: float, hub's longitude
        global_x_offset: float, rotation angle in degrees (0 = North, 90 = East)
        
    Returns:
        robots_lat_lon: numpy array of shape (2,n) containing [latitudes; longitudes]
    """
    # Create CRS objects for WGS84 and UTM
    wgs84 = CRS.from_epsg(4326)  # WGS84 geographic coordinate system
    utm = CRS.from_epsg(32618)   # UTM zone 18N
    
    # Create transformer objects (forward and inverse)
    transformer_to_utm = Transformer.from_crs(wgs84, utm, always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(utm, wgs84, always_xy=True)
    
    # Transform hub coordinates to UTM
    hub_x, hub_y = transformer_to_utm.transform(longitude_hub, latitude_hub)
    
    # Convert rotation angle to radians
    theta = np.radians(global_x_offset)
    
    # Create inverse rotation matrix
    R_inv = np.array([[np.cos(-theta), np.sin(-theta)],
                      [-np.sin(-theta), np.cos(-theta)]])
    
    # Un-rotate the coordinates
    unrotated_coords = R_inv @ robots_position
    
    # Add hub coordinates to get absolute UTM coordinates
    utm_x = unrotated_coords[0] + hub_x
    utm_y = unrotated_coords[1] + hub_y
    
    # Transform back to WGS84 (GPS coordinates)
    longitudes, latitudes = transformer_to_wgs84.transform(utm_x, utm_y)
    
    # Stack coordinates in the same format as input [lat; lon]
    robots_lat_lon = np.vstack((latitudes, longitudes))
    
    return robots_lat_lon

# Example usage
if __name__ == "__main__":
    # Example local coordinates (in meters)
    robots_position = np.array([
        [10, -20, 15, -5],    # x coordinates
        [30, -10, -25, 40]    # y coordinates
    ])
    
    # Hub GPS coordinates
    latitude_hub = 38.78753
    longitude_hub = -75.16214
    
    # Try different rotations
    rotations = [0, 90, 180, 270]
    
    for rotation in rotations:
        robots_lat_lon = local_to_gps_coordinates(
            robots_position, latitude_hub, longitude_hub, global_x_offset=rotation
        )
        
        print(f"\nGPS coordinates in WGS84 (rotation: {rotation}°):")
        print(f"Latitudes: {robots_lat_lon[0]}")
        print(f"Longitudes: {robots_lat_lon[1]}")
        
        # Verify the transformation by converting back to local coordinates
        local_coords = gps_to_local_coordinates(
            robots_lat_lon, latitude_hub, longitude_hub, global_x_offset=rotation
        )
        print("\nVerification - converting back to local coordinates:")
        print(f"Original X: {robots_position[0]}")
        print(f"Computed X: {local_coords[0]}")
        print(f"Original Y: {robots_position[1]}")
        print(f"Computed Y: {local_coords[1]}")
