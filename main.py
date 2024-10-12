# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 3D processing libraries
import open3d as o3d
import laspy

# Geospatial libraries
import rasterio
import alphashape as ash
import geopandas as gpd
import shapely as sh

from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import Polygon

# 2. Data Profiling

# Load the neighborhood point cloud (ensure lazrs or laszip is installed for .laz files)
try:
    las = laspy.read(r'neighborhood.laz')
except laspy.LaspyException as e:
    print(f"Error reading the LAS file: {e}")

# Display unique classifications in the dataset
print("Unique classifications in the dataset:", np.unique(las.classification))
# List available dimensions in the LAS file
print("Available dimensions in the LAS file:", [dimension.name for dimension in las.point_format.dimensions])

# Check and display CRS information if available
if len(las.vlrs) > 2:
    crs = las.vlrs[2].string
    print("Coordinate Reference System (CRS):", crs)
else:
    print("No CRS information available in VLRs.")

# 3. Data Pre-Processing

# 3.1. Building points initialization
# Create a mask to filter points classified as buildings (classification code 6)
pts_mask = las.classification == 2

# Extract coordinates of the filtered points
xyz_t = np.vstack((las.x[pts_mask], las.y[pts_mask], las.z[pts_mask]))

# Check if any points are found with the specified classification
if xyz_t.shape[1] > 0:
    # Convert to Open3D PointCloud and visualize
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(xyz_t.transpose())

    # Center the point cloud by translating it
    pcd_center = pcd_o3d.get_center()
    pcd_o3d.translate(-pcd_center)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd_o3d])
else:
    print("No points found with classification code 6.")
