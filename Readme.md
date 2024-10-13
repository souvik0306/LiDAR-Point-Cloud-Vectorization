# LiDAR Point Cloud Vectorization

In this project, I’m working on vectorizing LiDAR data using the [Vancouver dataset](https://opendata.vancouver.ca/explore/dataset/lidar-2022/map/?location=11,49.31483,-123.14438). The goal is to take raw 3D point cloud data, classify it, and convert it into useful vector shapes, such as building footprints and vegetation outlines.

## Classes

The LiDAR data includes the following classes:

- **0**: Bare-earth/low grass
- **1**: Low vegetation (<2m)
- **2**: High vegetation (>2m)
- **3**: Water
- **4**: Buildings
- **5**: Other
- **6**: Noise (outliers, errors)

## Overview

The process involves taking LiDAR data and identifying key elements like buildings, water bodies, and vegetation. After classifying the data, I vectorize it into 2D and 3D models. The project also focuses on filtering noise to improve the final output.

## Dataset

I’m using the Vancouver dataset, which provides rich LiDAR data for experimentation. It’s already classified into different features like ground, buildings, and vegetation, making it ideal for vectorization.

## Key Steps

1. **Preprocessing**: Set up the environment, load LiDAR data, and filter the points (e.g., removing noise).
2. **Vectorization**: Use the classified points to create 2D/3D models of buildings and other elements.
3. **Automation**: Automate the pipeline to process large datasets efficiently.

## Goals

- Classify and vectorize key features in the point cloud.
- Generate clean 3D models from the data.
- Filter out noise and improve data quality for better results.

## How to Use

### Environment Setup

Create a virtual environment (I use Conda) and install required libraries like `open3d`, `numpy`, and `laspy`.

### Running the Pipeline

1. Load the LiDAR dataset.
2. Classify the points.
3. Vectorize the results.
4. Visualize the 3D models with Open3D or export the results for further analysis.
