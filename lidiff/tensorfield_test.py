import torch
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import open3d as o3d

# Function to project points to a 2D depth image
def project_points_to_image(points, image_size=(224, 224), depth_bins=112):
    """
    Projects 3D points onto a 2D image plane to generate a depth image.

    Args:
        points (torch.Tensor): 3D points of shape [N, 3].
        image_size (tuple): Resolution of the depth image (width, height).
        depth_bins (int): Number of depth levels.

    Returns:
        torch.Tensor: A depth image of shape [H, W].
    """
    width, height = image_size

    # Normalize spatial coordinates (x, y) to image dimensions
    x = ((points[:, 0] + 1) / 2 * (width - 1)).long()  # Normalize to [0, width)
    y = ((points[:, 1] + 1) / 2 * (height - 1)).long()  # Normalize to [0, height)

    # Normalize depth (z) to [0, depth_bins)
    z = ((points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min()) * (depth_bins - 1)).long()

    # Initialize depth image
    depth_image = torch.zeros(height, width, dtype=torch.long, device=points.device)

    # Populate depth image (taking the nearest depth value per pixel)
    depth_image[y, x] = z
    return depth_image

# Load outdoor point cloud scene using Open3D
scene_path = "/nas2/jacob/LiDiff/lidiff/Datasets/test/000123.ply"  # Update with your file path
scene = o3d.io.read_point_cloud(scene_path)
scene = scene.voxel_down_sample(voxel_size=0.05)  # Downsample for faster processing
scene_points = torch.tensor(scene.points, dtype=torch.float32, device='cuda:1')

# Normalize scene points to [-1, 1] range
min_bounds = scene_points.min(dim=0)[0]
max_bounds = scene_points.max(dim=0)[0]
scene_points = 2 * (scene_points - min_bounds) / (max_bounds - min_bounds) - 1

# Add batch index to create x_feats
x_feats = torch.cat([torch.ones((scene_points.shape[0], 1), device='cuda:1'), scene_points], dim=1)  # Add batch index as the first column

# Quantized coordinates using MinkowskiEngine
x_coord = x_feats.clone()
resolution = 0.05
x_coord[:, 1:] = (x_coord[:, 1:] / resolution).round() * resolution  # Quantize spatial coordinates

# Create the TensorField
x_t = ME.TensorField(
    features=x_feats[:, 1:],  # Features
    coordinates=x_coord,  # Quantized coordinates
    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
    minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
    device='cuda:1',
)

# Extract quantized coordinates from the TensorField
tensorfield_coords = x_t.coordinates[:, 1:4]  # Exclude the batch index

# Generate depth images
original_image = project_points_to_image(scene_points, image_size=(224, 224), depth_bins=112)
quantized_image = project_points_to_image(tensorfield_coords, image_size=(224, 224), depth_bins=112)

# Save the depth images
plt.imsave('original_scene_image.png', original_image.cpu().numpy(), cmap='viridis')
plt.imsave('quantized_scene_image.png', quantized_image.cpu().numpy(), cmap='viridis')

# Output file paths
print("Original scene image saved to: original_scene_image.png")
print("Quantized scene image saved to: quantized_scene_image.png")
