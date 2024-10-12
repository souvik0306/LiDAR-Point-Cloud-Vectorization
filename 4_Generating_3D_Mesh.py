# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
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
pts_mask = las.classification == 6
xyz_t = np.vstack((las.x[pts_mask], las.y[pts_mask], las.z[pts_mask]))
pcd_o3d = o3d.geometry.PointCloud()
pcd_o3d.points = o3d.utility.Vector3dVector(xyz_t.transpose())
pcd_center = pcd_o3d.get_center()
pcd_o3d.translate(-pcd_center)
# o3d.visualization.draw_geometries([pcd_o3d])

# Ground Plane
pts_mask = las.classification == 2
xyz_t = np.vstack((las.x[pts_mask], las.y[pts_mask], las.z[pts_mask]))
ground_pts = o3d.geometry.PointCloud()
ground_pts.points = o3d.utility.Vector3dVector(xyz_t.transpose())
ground_pts.translate(-pcd_center)
# o3d.visualization.draw_geometries([ground_pts])

# Get Average Distance between building points
nn_distance = np.mean(pcd_o3d.compute_nearest_neighbor_distance())
print("Average distance between building points:", nn_distance)

# 4. Unsupervised Segmentation
epsilon = 2
min_cluster_size = 100
labels = np.array(pcd_o3d.cluster_dbscan(eps=epsilon, min_points=min_cluster_size, print_progress=True))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd_o3d.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd_o3d])

# 5. Selecting a segment
sel = 1
segment = pcd_o3d.select_by_index(np.where(labels == sel)[0])
# o3d.visualization.draw_geometries([segment])

# 6. Extracting the segment
points_2D = np.asarray(segment.points)[:, :2]
building_vector = ash.alphashape(points_2D, alpha=0.5)

# 7. Store in a geodataframe the 2D polygon
building_gdf = gpd.GeoDataFrame(geometry=[building_vector], crs='EPSG:26910')
print(building_gdf.head(1))

# 7.1 Computing semantics and attributes
# Height of the building
altitude = np.asarray(segment.points)[:, 2]+pcd_center[2]
height_test = np.max(altitude) - np.min(altitude)
print("Height of the building (Initial):", height_test)

# Define Ground Plane
query_point = segment.get_center()
query_point[2] = segment.get_min_bound()[2]
pcd_tree = o3d.geometry.KDTreeFlann(ground_pts)
[k, idx, _] = pcd_tree.search_knn_vector_3d(query_point, 200)

sample = ground_pts.select_by_index(idx, invert=False)
sample.paint_uniform_color([0.5, 0.5, 0.5])
# o3d.visualization.draw_geometries([sample, ground_pts])

ground_zero = sample.get_center()[2]
height = segment.get_max_bound()[2] - ground_zero
print("Height of the building (Final):", height)
print("Difference in height:", height - height_test)

# 7.2 Computing parameters
building_gdf[['id']] = sel
building_gdf[['height']] = height
building_gdf[['area']] = building_vector.area
building_gdf[['perimeter']] = building_vector.length
building_gdf[['local_cx', 'local_cy', 'local_cz']] = np.asarray([building_vector.centroid.x, building_vector.centroid.y, ground_zero])
building_gdf[['transl_x', 'transl_y', 'transl_z']] = pcd_center
building_gdf[['pts_number']] = len(segment.points)
print(building_gdf.head(1))

# 8. 2D to 3D 
# Generate the vertice list 
vertices = list(building_vector.exterior.coords)

# Construct the Open3D Object
polygon_2d = o3d.geometry.LineSet()
polygon_2d.points = o3d.utility.Vector3dVector([point + (0,) for point in vertices])
polygon_2d.lines = o3d.utility.Vector2iVector([[i, (i + 1) % len(vertices)] for i in range(len(vertices))])
# o3d. visualization.draw_geometries([polygon_2d])

# 8.2 The top layer
extrusion = o3d.geometry.LineSet()
extrusion.points = o3d.utility.Vector3dVector([point + (height,) for point in vertices])
extrusion.lines = o3d.utility.Vector2iVector([[i, (i + 1) % len(vertices)] for i in range(len(vertices))])
# o3d. visualization.draw_geometries([polygon_2d, extrusion])

# Plot the vertices
temp = polygon_2d + extrusion
temp.points
temp_o3d = o3d.geometry.PointCloud()
temp_o3d.points = temp.points
# o3d.visualization.draw_geometries([temp_o3d])

# 9. Generating the base vertices for the 3D mesh
a = np.array(building_vector.exterior.coords)
b = np.ones([a.shape[0],1])*sample.get_center()[2]
c = np.ones([a.shape[0],1])*(sample.get_center()[2] + height)

ground_pc = np.hstack((a,b))
up_pc = np.hstack((a,c))

temp_o3d = o3d.geometry.PointCloud()
temp_o3d.points = o3d.utility.Vector3dVector(np.concatenate((ground_pc, up_pc), axis=0))
# o3d.visualization.draw_geometries([temp_o3d])

# 10. Computing the alpha shapre of the 3D base points
alpha = 20
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(temp_o3d, alpha)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.5, 0.4, 0])
# o3d.visualization.draw_geometries([temp_o3d, mesh, segment], mesh_show_back_face=True)

# 11. Post-Porcessing Operations and Export ˀ
mesh.translate(pcd_center)
o3d.io.write_triangle_mesh("building_mesh.ply", mesh, write_ascii=True, compressed=True, write_vertex_normals=False, write_vertex_colors=False, write_triangle_uvs=False)
building_gdf.to_file("building_gdf.shp")
