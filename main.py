# Base libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 3D Libraries
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

# neighborhood point cloud (Make sure lazrs or laszip is installed for .laz decompression)
try:
    las = laspy.read(r'neighborhood.laz')
except laspy.LaspyException as e:
    print(f"Error reading the LAS file: {e}")

# explore the classification field
print("Unique classifications in the dataset:", np.unique(las.classification))
print("Available dimensions in the LAS file:", [dimension.name for dimension in las.point_format.dimensions])

# explore CRS info (Check if CRS exists, it could fail in some datasets)
if len(las.vlrs) > 2:
    crs = las.vlrs[2].string
    print("Coordinate Reference System (CRS):", crs)
else:
    print("No CRS information available in VLRs.")

# 3. Data Pre-Processing

# 3.1. Building points initialization
# Create a Mask to filter points with classification code 6 (buildings)
pts_mask = las.classification == 6

# Apply the mask and get the coordinates of the filtered dataset (note the t for transpose)
xyz_t = np.vstack((las.x[pts_mask], las.y[pts_mask], las.z[pts_mask]))

# Check if the mask returns any points
if xyz_t.shape[1] > 0:
    # Transform to Open3D PointCloud and visualize
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(xyz_t.transpose())

    # Translate the point cloud to center it
    pcd_center = pcd_o3d.get_center()
    pcd_o3d.translate(-pcd_center)

    # Reduce the number of points (downsample)
    downsampled_pcd = pcd_o3d.voxel_down_sample(voxel_size=1.0)

    # Visualize the downsampled point cloud
    o3d.visualization.draw_geometries([downsampled_pcd], window_name="Downsampled Point Cloud")

else:
    print("No points found with classification code 6.")
