
import os
import numpy as np

from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
from sklearn.cluster import KMeans

# Function to check if a directory exists, and create it if it doesn't
def is_dir_path(string):
    if not os.path.isdir(string):
        os.makedirs(string, exist_ok=True) # Create the directory if it doesn't exist
    return string

# Function to segment the lung from a CT image using KMeans and morphological operations
def segment_lung(img):
    # Normalize the image by subtracting the mean and dividing by the standard deviation
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std

    # Extract the middle part of the image to estimate threshold
    middle = img[100:400, 100:400]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)

    # Replace max and min values with the mean to avoid artifacts
    img[img == max] = mean
    img[img == min] = mean

    # Apply median filter to smooth the image
    img = median_filter(img, size=3)
    # Apply anisotropic diffusion to preserve edges while smoothing
    img = anisotropic_diffusion(img)

    # Perform KMeans clustering to segment the image into two regions
    kmeans = KMeans(n_clusters = 2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers) # Calculate the threshold as the average of cluster centers
    # Create a binary mask based on the threshold
    thresh_img = np.where(img < threshold, 1.0, 0.0)

    # Apply morphological operations: erosion followed by dilation
    eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
    dilation = morphology.dilation(eroded, np.ones([10, 10]))

    # Label the connected components in the image
    labels = measure.label(dilation)
    label_vals = np.unique(labels) # Get unique labels in the binary image
    regions = measure.regionprops(labels) # Get properties of laveled regions
    good_labels = []
    # Filter out small regions based on their bounding box size
    for prop in regions:
        B = prop.bbox # Get the bounding box of the region
        # Keep regions that are within a reasonable size range
        if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)

    # Create an empty mask and add the selected regions to it
    mask = np.ndarray([512, 512], dtype=np.int8)
    mask[:] = 0 # Initialize the mask with zeros

    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)# Add selected labels to the mask

    # Apply dilation to the mask to smooth the boundaries
    mask = morphology.dilation(mask, np.ones([10, 10]))

    # Return the final mask multiplied by the original image ( to apply the mask to the image)
    return mask*img

# Function to count the number of trainable parameters in a PyTorch model
def count_params(model):
    return sum(p.numel() for p in model.parameters if p.requires_grad) # Sum the number of parameters that require gradients