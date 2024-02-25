# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from CustomDataset import Genki4kDataset_Labeless
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import os
import shutil
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances



# Set up the directory where the images are stored
data_dir = 'genki4k/files'

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert images to PyTorch tensors
])

# Initialize the dataset without labels
dataset = Genki4kDataset_Labeless(data_dir, transform=transform)

# Create a DataLoader to iterate over the dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load the FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Extract features
features = []
with torch.no_grad():
    for images in dataloader:
        images = images.to(device)
        embeddings = model(images)
        features.append(embeddings.cpu().numpy())

features = np.concatenate(features, axis=0)


# Perform Spectral Clustering
n_clusters = 2  # Adjust the number of clusters as needed
spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
cluster_labels = spectral.fit_predict(features)

# Use t-SNE for visualization of high-dimensional data
tsne = TSNE(n_components=2, random_state=42)
reduced_features = tsne.fit_transform(features)

# Plotting the clustered data
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis')
plt.title('Spectral Clustering Results')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.colorbar(scatter, label='Cluster Label')
plt.show()

# Select random samples from each cluster
samples_per_cluster = 5  # Adjust the number of samples to display per cluster
clustered_images = {}

for cluster_num in range(n_clusters):
    # Get indexes of images in this cluster
    indexes = np.where(cluster_labels == cluster_num)[0]
    # Randomly select a few indexes for display
    selected_indexes = np.random.choice(indexes, samples_per_cluster, replace=False)
    # Retrieve the selected images using the dataset object
    clustered_images[cluster_num] = [dataset[i] for i in selected_indexes]

# Now, visualize the selected samples
fig, axes = plt.subplots(n_clusters, samples_per_cluster, figsize=(20, 8))

for cluster_num, images in clustered_images.items():
    for i, image_tensor in enumerate(images):
        ax = axes[cluster_num, i]
        # Convert PyTorch tensor to numpy array and change axis order from CxHxW to HxWxC
        image_np = image_tensor.numpy().transpose(1, 2, 0)
        ax.imshow(image_np)  # Now the image is in the correct format
        ax.axis('off')
        ax.set_title(f"Cluster {cluster_num} Sample {i+1}")

plt.tight_layout()
plt.show()

# Find representative images for each cluster
representative_imgs_indices = []
for cluster_num in range(n_clusters):
    cluster_indices = np.where(cluster_labels == cluster_num)[0]
    cluster_features = features[cluster_indices]
    # Compute the cluster center by averaging the features
    cluster_center = cluster_features.mean(axis=0)
    # Compute distances of the cluster's points to the cluster center
    distances = pairwise_distances(cluster_features, [cluster_center])
    # Find the index of the minimum distance
    representative_img_index = cluster_indices[distances.argmin()]
    representative_imgs_indices.append(representative_img_index)

# Visualize the representative images
fig, axes = plt.subplots(1, n_clusters, figsize=(15, 5))
for ax, img_idx in zip(axes, representative_imgs_indices):
    img_tensor = dataset[img_idx]  # Retrieve the image tensor from the dataset
    # Make sure the tensor is in CPU memory and detach it from the graph
    img_tensor = img_tensor.cpu().detach()
    # Normalize to [0, 255] and convert to uint8
    img_np = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    # Convert the numpy array to a PIL Image for display
    img = Image.fromarray(img_np)
    ax.imshow(img)
    ax.axis('off')  # Hide the axes ticks
    ax.set_title(f"Cluster {cluster_labels[img_idx]}")  # Set the title to the cluster number

plt.tight_layout()
plt.show()








