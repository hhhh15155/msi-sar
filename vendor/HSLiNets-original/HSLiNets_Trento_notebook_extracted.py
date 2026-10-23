# ---- cell ----
#pip install spectral mat73  einops
# ---- cell ----
import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

import sys

import os

import math

import random



from scipy.ndimage import rotate



from einops import rearrange

import torch.optim as optim

import torchvision

from torchvision import datasets, transforms

from scipy import io

import torch.utils.data

import scipy.io as sio

import mat73

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader



from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split



import torch.optim as optim

from torch.utils.data import DataLoader

import copy

import time

import psutil

import os



from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

random.seed(42)

np.random.seed(42)



m=1
# ---- cell ----
# from google.colab import drive

# drive.mount('/content/drive')
# ---- cell ----
#! ls '/content/drive/MyDrive/A02_RemoteSensingData/TrentoDataSet/'
# ---- cell ----
# Define the path

#path='/content/drive/MyDrive/A02_RemoteSensingData/TrentoDataSet/'
# ---- cell ----
# 2.1 Define the class information

class_info = [

    (1, "Apple trees", 129, 3905, 4034),

    (2, "Buildings ", 125, 2778, 2903),

    (3, "Ground", 105, 374, 479),

    (4, "Wood", 154, 8969, 9123),

    (5, "Vineyard", 184, 10317, 10501),  # Corrected this line

    (6, "Roads", 122, 3052, 3174)

]



class_dict = {class_number: {"class_name": class_name, "training": training, "test": test, "samples": samples} for class_number, class_name, training, test, samples in class_info}



# for class_number, class_info in class_dict.items():

#     print(f"Class {class_number}: {class_info['class_name']}")

#     print(f"Training Samples: {class_info['training']}")

#     print(f"Test Samples: {class_info['test']}")

#     print(f"Total Samples: {class_info['samples']}")

#     print()



#print(class_dict)

# path

path ='/content/drive/MyDrive/A02_RemoteSensingData/TrentoDataSet/'

# Loader HSI TR_map_2018

trento_data=sio.loadmat('trento_data.mat')['HSI_data']

trento_data.shape
# ---- cell ----
## Define the path

#path='/content/drive/MyDrive/A02_RemoteSensingData/TrentoDataSet/'



# Loader HSI TR_map_2018

trento_data=sio.loadmat('trento_data.mat')



# Extract the HSI_data and LiDAR_data arrays

trento_hsi_data = trento_data['HSI_data']

trento_lidar_data = trento_data['LiDAR_data']

print('trento_lidar_data shape:', trento_lidar_data.shape)

# Reshape the data

trento_lidar_data = np.reshape(trento_lidar_data, (166, 600, 1))



trento_gt=trento_data['ground']

print('trento_hsi_data shape:', trento_hsi_data.shape)

print('trento_lidar_data shape:', trento_lidar_data.shape)

print('trento_gt shape:', trento_gt.shape)

# ---- cell ----
# # Raw Data Normalisation

# import numpy as np

# from sklearn.preprocessing import MinMaxScaler, StandardScaler



# # Normalization function for HSI and LiDAR data

# def normalize_data(data, method="minmax"):

#     """

#     Normalize hyperspectral or LiDAR data.

    

#     Args:

#         data (numpy array): Input data array (e.g., HSI or LiDAR).

#         method (str): Normalization method, "minmax" or "zscore".

    

#     Returns:

#         numpy array: Normalized data.

#     """

#     if method == "minmax":

#         # Reshape the data for MinMaxScaler

#         scaler = MinMaxScaler()

#         flat_data = data.reshape(-1, data.shape[-1])

#         normalized_data = scaler.fit_transform(flat_data).reshape(data.shape)

#     elif method == "zscore":

#         # Standard normalization: zero mean, unit variance

#         scaler = StandardScaler()

#         flat_data = data.reshape(-1, data.shape[-1])

#         normalized_data = scaler.fit_transform(flat_data).reshape(data.shape)

#     else:

#         raise ValueError("Unsupported normalization method. Use 'minmax' or 'zscore'.")

    

#     return normalized_data



# # Load hyperspectral data

# trento_hsi_data = trento_data['HSI_data']

# print('HSI data shape:', trento_hsi_data.shape)



# # Normalize HSI data

# trento_hsi_data_normalized = normalize_data(trento_hsi_data, method="minmax")

# print('HSI normalized shape:', trento_hsi_data_normalized.shape)



# # Load LiDAR data

# trento_lidar_data = trento_data['LiDAR_data']

# print('LiDAR data shape:', trento_lidar_data.shape)



# # Normalize LiDAR data

# # Since LiDAR has a single band, we normalize over the first two dimensions only

# lidar_min = np.min(trento_lidar_data )

# lidar_max = np.max(trento_lidar_data )

# trento_lidar_data_normalized = (trento_lidar_data  - lidar_min) / (lidar_max - lidar_min)

# trento_lidar_data_normalized  = np.reshape(trento_lidar_data_normalized , (166, 600, 1))

# print('LiDAR normalized shape:', trento_lidar_data_normalized.shape)
# ---- cell ----
trento_hsi_data=trento_hsi_data_normalized

trento_lidar_data=trento_lidar_data_normalized
# ---- cell ----
# # Set the background class to 0

# Define patch size and stride

patch_size = m

stride = 1



# Create an empty list to store patches and labels

hsi_samples = []

lidar_samples = []

labels = []



# Initialize a dictionary to store class count

class_count = {i: 0 for i in class_dict.keys()}



# Function to check if all classes have the required number of samples

def all_classes_completed(class_count, class_dict):

    return all(class_count[class_num] == class_dict[class_num]["samples"] for class_num in class_dict.keys())



while not all_classes_completed(class_count, class_dict):

    # Loop through the ground truth data

    for label in class_dict.keys():

        # Get the coordinates of the ground truth pixels

        #coords = np.argwhere((trento_gt == label) & (mask > 0))

        coords = np.argwhere(trento_gt == label)



        # Shuffle the coordinates to randomize the patch extraction

        np.random.shuffle(coords)



        for coord in coords:

            i, j = coord

            # Calculate the patch indices

            i_start, i_end = i - patch_size // 2, i + patch_size // 2 + 1

            j_start, j_end = j - patch_size // 2, j + patch_size // 2 + 1



            # Check if the indices are within the bounds of the HSI data

            if i_start >= 0 and i_end <= trento_hsi_data.shape[0] and j_start >= 0 and j_end <= trento_hsi_data.shape[1]:

                # Extract the patch

                hsi_patch = trento_hsi_data[i_start:i_end, j_start:j_end, :]



                # Extract the LiDAR patch

                lidar_patch = trento_lidar_data[i_start:i_end, j_start:j_end]



                # If the class count is less than the required samples

                if class_count[label] < class_dict[label]["samples"]:

                    # Append the patch and its label to the list

                    hsi_samples.append(hsi_patch)

                    lidar_samples.append(lidar_patch)

                    labels.append(label)

                    class_count[label] += 1



                    # If all classes have the required number of samples, exit the loop

                    if all_classes_completed(class_count, class_dict):

                        break



# Convert the list of patches and labels into arrays

hsi_samples = np.array(hsi_samples)

lidar_samples = np.array(lidar_samples)

labels = np.array(labels)

print('hsi_samples shape:', hsi_samples.shape)

print('lidar_samples shape:', lidar_samples.shape)

print('labels shape:', labels.shape)
# ---- cell ----
import numpy as np



# Assuming `hsi_samples`, `lidar_samples`, and `labels` are defined as numpy arrays

# Assuming `training_samples_dict` is defined and contains the desired number of training samples for each label

# Create training_samples_dict based on class_dict

training_samples_dict = {class_num: class_info["training"] for class_num, class_info in class_dict.items()}

hsi_samples = np.array(hsi_samples)

lidar_samples = np.array(lidar_samples)

labels = np.array(labels)



# Initialize arrays to hold indices for training and test sets

train_indices = np.array([], dtype=int)

test_indices = np.array([], dtype=int)



# Split indices into training and test sets

for label, train_samples in training_samples_dict.items():

    # Get indices of the current class

    class_indices = np.where(labels == label)[0]



    # Randomly shuffle the indices

    np.random.shuffle(class_indices)



    # Split the indices into training and test set indices

    train_idx = class_indices[:train_samples]

    test_idx = class_indices[train_samples:]



    # Append to the overall training and test indices

    train_indices = np.concatenate([train_indices, train_idx])

    test_indices = np.concatenate([test_indices, test_idx])



# Use the indices to split the samples into training and test sets

hsi_train_samples = hsi_samples[train_indices]

lidar_train_samples = lidar_samples[train_indices]

training_labels = labels[train_indices]



hsi_test_samples = hsi_samples[test_indices]

lidar_test_samples = lidar_samples[test_indices]

test_labels = labels[test_indices]



# Print shapes to verify

print('hsi_training_samples shape:', hsi_train_samples.shape)

print('lidar_training_samples shape:', lidar_train_samples.shape)

print('training_labels shape:', training_labels.shape)

print('hsi_test_samples shape:', hsi_test_samples.shape)

print('lidar_test_samples shape:', lidar_test_samples.shape)

print('test_labels shape:', test_labels.shape)
# ---- cell ----
# Generate a list of indices for the 144 bands

full_bands = list(range(63))



# Print the list of band indices

print("Band Indices:", full_bands)

band_indices=full_bands





# Extract the corresponding bands from the augmented training samples

selected_hsi_train_samples = hsi_train_samples[:,:,:, band_indices]

selected_hsi_test_samples = hsi_test_samples[:,:,:, band_indices]

selected_lidar_train =  lidar_train_samples

selected_lidar_test = lidar_test_samples

print('selected_hsi_train_samples  shape:', selected_hsi_train_samples .shape)

print('selected_hsi_test_sample shape:', selected_hsi_test_samples.shape)

print('selected_lidar_train shape:', selected_lidar_train.shape)

print('selected_lidar_test shape:', selected_lidar_test.shape)



# #Adding LiDAR Data to Reduced bands HSI Data,concatenate the data along axis 3

# Concatenate along axis 3 (the band/channel axis)

selected_hsi_lidar_train_samples = np.concatenate((selected_hsi_train_samples, selected_lidar_train), axis=3)

selected_hsi_lidar_test_samples = np.concatenate((selected_hsi_test_samples, selected_lidar_test), axis=3)



# Check the shapes of the concatenated samples

print('selected_hsi_lidar_train_samples shape:', selected_hsi_lidar_train_samples.shape)

print('hsselected_hsi_lidar_test_samples shape:', selected_hsi_lidar_test_samples.shape)





X_train=selected_hsi_lidar_train_samples

X_test=selected_hsi_lidar_test_samples

y_train=training_labels

y_test=test_labels



print('X_train shape:', X_train.shape)

print('X_test shape:', X_test.shape)

print('y_train shape:', y_train.shape)

print('y_test shape:', y_test.shape)
# ---- cell ----
# import numpy as np

# from scipy.ndimage import rotate



def augment_training_data(hsi_training_data, training_labels, rotations=[45, 90, 135], flip_up_down=True, flip_left_right=True):

    augmented_hsi = []

    augmented_labels = []



    for hsi, label in zip(hsi_training_data, training_labels):

        # Original data

        augmented_hsi.append(hsi)

        augmented_labels.append(label)



        # Rotations

        for angle in rotations:

            hsi_rotated = rotate(hsi, angle, axes=(0, 1), reshape=False, mode='nearest')

            augmented_hsi.append(hsi_rotated)

            augmented_labels.append(label)



        # Flip up-down

        if flip_up_down:

            hsi_flipped_ud = np.flipud(hsi)

            augmented_hsi.append(hsi_flipped_ud)

            augmented_labels.append(label)



        # Flip left-right

        if flip_left_right:

            hsi_flipped_lr = np.flipud(hsi)

            augmented_hsi.append(hsi_flipped_lr)

            augmented_labels.append(label)



    return np.array(augmented_hsi), np.array(augmented_labels)



# Augmenting the training samples

augmented_hsi_training_samples, augmented_training_labels = augment_training_data(X_train,y_train)



# Print shapes to verify the augmented training data

print('Augmented HSI training samples shape:', augmented_hsi_training_samples.shape)

print('Augmented training labels shape:', augmented_training_labels.shape)
# ---- cell ----
# import numpy as np

# import torch

# from torch.utils.data import TensorDataset, DataLoader

# from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import StandardScaler



# Split the augmented training data into training and validation sets

X_train, X_val, y_train, y_val = train_test_split(

    augmented_hsi_training_samples, augmented_training_labels, test_size=0.1, random_state=42, stratify=augmented_training_labels

)



print('X_train shape:', X_train.shape)

print('X_val shape:', X_val.shape)

print('y_train shape:', y_train.shape)

print('y_val shape:', y_val.shape)



print('X_test shape:', X_test.shape)

print('y_test shape:', y_test.shape)



# Store the data shapes for model

original_train_shape = X_train.shape  # (samples, height, width, channels)

original_val_shape = X_val.shape

original_test_shape = X_test.shape



# Reshape for standardization

X_train_reshaped = X_train.reshape(-1, original_train_shape[1]*original_train_shape[2]*original_train_shape[3])

X_val_reshaped = X_val.reshape(-1, original_val_shape[1]*original_val_shape[2]*original_val_shape[3])

X_test_reshaped = X_test.reshape(-1, original_test_shape[1]*original_test_shape[2]*original_test_shape[3])



# Standardize

scaler = StandardScaler()

X_train_standardized = scaler.fit_transform(X_train_reshaped)

X_val_standardized = scaler.transform(X_val_reshaped)

X_test_standardized = scaler.transform(X_test_reshaped)



# Reshape back to original 4D shape

X_train = X_train_standardized.reshape(original_train_shape)

X_val = X_val_standardized.reshape(original_val_shape)

X_test = X_test_standardized.reshape(original_test_shape)



# Check the reshaped arrays

print("After standardization:")

print("X_train shape:", X_train.shape)

print("X_val shape:", X_val.shape)

print("X_test shape:", X_test.shape)



# Convert the split datasets to tensor datasets

train_dataset = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train).long())

val_dataset = TensorDataset(torch.tensor(X_val.astype(np.float32)), torch.tensor(y_val).long())

test_dataset = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test).long())



# Create DataLoader instances for training, validation, and testing

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# ---- cell ----
# Configuration class

class Config:

    def __init__(self, in_channels, num_patches, kernel_size, patch_size, emb_size, dim, depth, heads, dim_head, mlp_ratio, num_classes, dropout, pos_emb_size, class_emb_size, stride, output_dim):  # Set default output_dim to 1

        self.in_channels = in_channels

        self.num_patches = num_patches

        self.kernel_size = kernel_size

        self.patch_size = patch_size

        self.emb_size = emb_size

        self.dim = dim

        self.depth = depth

        self.heads = heads

        self.dim_head = dim_head

        self.mlp_ratio = mlp_ratio

        self.num_classes = num_classes

        self.dropout = dropout

        self.pos_emb_size = pos_emb_size

        self.class_emb_size = class_emb_size

        self.stride = stride

        self.output_dim = output_dim  # Ensure output_dim is a part of the config
# ---- cell ----
# Reversed along bands  Method 2



class HSIVimBlock(nn.Module):

    def __init__(self, spatial_dim, num_bands, hidden_dim, output_dim, delta_param_init):

        super(HSIVimBlock, self).__init__()

        self.spatial_dim = spatial_dim

        self.num_bands = num_bands

        self.hidden_dim = hidden_dim



        # LayerNorm over Bands

        self.norm = nn.LayerNorm(num_bands)



        # Adjusted linear layers to handle reversed bands

        self.linear_x = nn.Linear(num_bands, hidden_dim)

        self.linear_z = nn.Linear(num_bands, hidden_dim)



        # Convolution for forward and backward paths

        self.forward_conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        self.backward_conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)



        # Trainable matrices and delta parameter

        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.B = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.delta_param = nn.Parameter(torch.full((hidden_dim,), delta_param_init))



        # Linear layers for final transformation

        self.linear_forward = nn.Linear(hidden_dim, output_dim)

        self.linear_backward = nn.Linear(hidden_dim, output_dim)



    def forward(self, x):

        Batch, H, W, Bands = x.shape  # Input shape: [Batch, Height, Width, Bands]



        # Reverse bands for z_proj

        x_reversed = torch.flip(x, dims=[-1])  # Reverse along the Bands dimension



        # Flatten spatial dimensions for processing

        x = x.view(Batch, H * W, Bands)  # Shape: [Batch, H*W, Bands]

        z = x_reversed.view(Batch, H * W, Bands)  # Shape: [Batch, H*W, Bands]



        # Apply LayerNorm to the Bands dimension

        x = self.norm(x)  # Shape: [Batch, H*W, Bands]

        z = self.norm(z)  # Shape: [Batch, H*W, Bands]



        # Linear projection to hidden dimensions

        x_proj = self.linear_x(x)  # Shape: [Batch, H*W, hidden_dim]

        z_proj = self.linear_z(z)  # Shape: [Batch, H*W, hidden_dim]



        # Reshape for Conv1d compatibility

        x_proj = x_proj.permute(0, 2, 1)  # Shape: [Batch, hidden_dim, H*W]

        z_proj = z_proj.permute(0, 2, 1)  # Shape: [Batch, hidden_dim, H*W]



        # Reverse the spatial dimension for backward path

        z_proj_reversed = torch.flip(z_proj, dims=[-1])  # Reverse along the spatial dimension



        # Conv1D processing

        x_forward = F.silu(self.forward_conv1d(x_proj))

        x_backward = F.silu(self.backward_conv1d(z_proj_reversed))



        # Apply delta parameter

        delta_expanded = self.delta_param.unsqueeze(0).unsqueeze(2)  # Shape: [1, hidden_dim, 1]



        # Forward and backward SSM processing

        forward_ssm_output = torch.tanh(

            self.forward_conv1d(x_proj) + torch.matmul(self.A, x_proj) + delta_expanded

        )

        backward_ssm_output = torch.tanh(

            self.backward_conv1d(z_proj_reversed) + torch.matmul(self.B, z_proj_reversed) + delta_expanded

        )



        # Combine forward and backward outputs

        forward_reduced = forward_ssm_output.mean(dim=2)  # Shape: [Batch, hidden_dim]

        backward_reduced = backward_ssm_output.mean(dim=2)  # Shape: [Batch, hidden_dim]



        # Linear layers for final transformation

        y_forward = self.linear_forward(forward_reduced)  # Shape: [Batch, output_dim]

        y_backward = self.linear_backward(backward_reduced)  # Shape: [Batch, output_dim]



        # Combine forward and backward results

        y_combined = y_forward + y_backward  # Shape: [Batch, output_dim]



        return y_combined
# ---- cell ----
# # Version 2.0 This involves reversing the input tensor for the backward path before applying the backward_conv1d operation



# class HSIVimBlock(nn.Module):

#     def __init__(self, spatial_dim, num_bands, hidden_dim, output_dim, delta_param_init):

#         super(HSIVimBlock, self).__init__()

#         input_dim = spatial_dim * spatial_dim * num_bands

#         self.norm = nn.LayerNorm(input_dim)

#         # Initialization with self.hidden_dim

#         self.spatial_dim = spatial_dim

#         self.num_bands = num_bands

#         self.hidden_dim = hidden_dim



#         # LayerNorm is now expecting a flattened feature vector of Bands*H*W elements

#         self.norm = nn.LayerNorm(num_bands * spatial_dim * spatial_dim)



#         # Adjust linear layers according to the new input dimension

#         self.linear_x = nn.Linear(num_bands * spatial_dim * spatial_dim, hidden_dim)

#         self.linear_z = nn.Linear(num_bands * spatial_dim * spatial_dim, hidden_dim)



#         self.forward_conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

#         self.backward_conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)



#         self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

#         self.B = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

#         #self.C = nn.Parameter(torch.randn(output_dim, hidden_dim))

#         self.delta_param = nn.Parameter(torch.full((hidden_dim,), delta_param_init))



#         self.linear_forward = nn.Linear(hidden_dim, output_dim)

#         self.linear_backward = nn.Linear(hidden_dim, output_dim)



#     def forward(self, x):

#         Batch, H, W, Bands = x.shape  # Correct shape extraction assuming [Batch, Height, Width, Bands]



#         # Correctly reshape for LayerNorm to flatten all spatial and spectral information

#         x = x.reshape(Batch, -1)  # New shape: [Batch, Bands*H*W]



#         # Normalize across the flattened spatial-spectral data

#         x = self.norm(x)



#         # Projection to hidden dimensions

#         x_proj = self.linear_x(x)

#         z_proj = self.linear_z(x)



#         # Ensure correct reshaping for Conv1d compatibility

#         x_proj = x_proj.view(Batch, self.hidden_dim, -1)

#         z_proj = z_proj.view(Batch, self.hidden_dim, -1)



#         # Reverse z_proj for the backward path

#         z_proj_reversed = torch.flip(z_proj, dims=[-1])



#         # Bidirectional Conv1d processing using reversed input for the backward path

#         x_forward = F.silu(self.forward_conv1d(x_proj))

#         x_backward = F.silu(self.backward_conv1d(z_proj_reversed))



#         # Apply delta parameter correctly

#         delta_expanded = self.delta_param.unsqueeze(0).unsqueeze(2)  # Correct shape for broadcasting



#         # SSM processing with delta applied, using the original and reversed inputs for forward and backward paths respectively

#         forward_ssm_output = torch.tanh(self.forward_conv1d(x_proj) + self.A * delta_expanded)

#         backward_ssm_output = torch.tanh(self.backward_conv1d(z_proj_reversed) + self.B * delta_expanded)



#         # Combine forward and backward outputs into a single representation

#         forward_reduced = forward_ssm_output.mean(dim=2)

#         backward_reduced = backward_ssm_output.mean(dim=2)



#         # Combine the reduced forward and backward paths

#         y_forward = self.linear_forward(forward_reduced)

#         y_backward = self.linear_backward(backward_reduced)



#         # Element-wise sum of forward and backward outputs

#         y_combined = y_forward + y_backward



#         # Return the combined output

#         return y_combined

# ---- cell ----
# New version

# import torch

# import torch.nn as nn

# import torch.nn.functional as F

class SpatialFeatureProcessing(nn.Module):

    def __init__(self, input_channels):

        super(SpatialFeatureProcessing, self).__init__()

        self.conv_layers = nn.Sequential(

            # First convolutional layer with dilation rate of 1 (standard convolution)

            nn.Conv2d(in_channels=input_channels, out_channels=256, kernel_size=(3, 3), padding=1, dilation=1),

            nn.ReLU(),

            nn.BatchNorm2d(256),

            # Second convolutional layer with a higher dilation rate to increase the receptive field

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=2, dilation=2),  # Note the increased padding to maintain the spatial dimensions

            nn.ReLU(),

            nn.BatchNorm2d(512)

        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Adding global average pooling



    def forward(self, x):

        x = self.conv_layers(x)

        x = self.global_avg_pool(x)  # Apply global average pooling

        x = torch.flatten(x, start_dim=1)  # Flatten all dimensions except batch

        return x

# ---- cell ----
class Classifier(nn.Module):

    def __init__(self, in_features, num_classes):

        super(Classifier, self).__init__()

        self.fc_layers = nn.Sequential(

            nn.Linear(in_features=in_features, out_features=1024),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(in_features=1024, out_features=num_classes),

        )



    def forward(self, x):

        x = self.fc_layers(x)

        # Remove softmax here if you're using a loss function that includes it, such as nn.CrossEntropyLoss

        return x

# ---- cell ----
class HSIClassificationMambaModel(nn.Module):

    def __init__(self, spatial_dim, num_bands, hidden_dim, output_dim, delta_param_init, num_classes):

        super(HSIClassificationMambaModel, self).__init__()

        self.vim_block = HSIVimBlock(spatial_dim, num_bands, hidden_dim, output_dim, delta_param_init)

        self.output_dim = output_dim  # Save output_dim as an attribute of the class



        # Initialize SpatialFeatureProcessing and Classifier here

        # Adjusted to pass 'output_dim' as 'input_channels' to SpatialFeatureProcessing

        self.spatial_processing = SpatialFeatureProcessing(input_channels=output_dim)

        # Assuming the output of SpatialFeatureProcessing matches the in_features expected by Classifier

        self.classifier = Classifier(in_features=512, num_classes=num_classes)



    def forward(self, x):

        x = self.vim_block(x)

        # This is a placeholder. Actual reshaping depends on the output of HSIVimBlock and the input expectation of SpatialFeatureProcessing

        x = x.view(-1, self.output_dim, 1, 1)  # Reshape to include spatial dimensions if needed

        x = self.spatial_processing(x)



        # Flatten the output from spatial processing if it's not already flat

        x = torch.flatten(x, start_dim=1)



        x = self.classifier(x)

        return x
# ---- cell ----
# # At The MEarly Fusion Stage , Lidar as additional band

# # Instantiate the model

# model = HSIClassificationMambaModel(

#     spatial_dim=3,

#     num_bands=64,

#     hidden_dim=512,

#     output_dim=256,

#     delta_param_init=0.01,

#     num_classes=6

# )



# # Print the model architecture

# print(model)
# ---- cell ----
# # At The Middel Stage

# # Instantiate the model

# model = HSILidClassificationMambaModel(

#     spatial_dim=7,

#     hsi_num_bands=63,

#     lidar_num_bands=1,

#     hidden_dim=512,

#     output_dim=256,

#     delta_param_init=0.01,

#     num_classes=20

# )



# # Print the model architecture

# print(model)
# ---- cell ----
# Before the training loop, to record the initial memory usage (GPU)

if torch.cuda.is_available():

    torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats at the start

    initial_memory = torch.cuda.memory_allocated()

    print(f"Initial Memory Allocated: {initial_memory / 1e6} MB")





# 1.0 Training Model in GPU for HSI loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#model = HSIClassificationMambaModel(

#    spatial_dim=7, num_bands=145, hidden_dim=256, output_dim=128, delta_param_init=0.01, num_classes=15

#)



model = HSIClassificationMambaModel(

    spatial_dim=m, num_bands=64, hidden_dim=512, output_dim=256, delta_param_init=0.01, num_classes=6

)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.000001)



epochs = 100

best_val_loss = float('inf')

best_model_wts = copy.deepcopy(model.state_dict())

patience = 30



start_time = time.time()  # Step 2: Record the start time



for epoch in range(epochs):

    model.train()

    running_train_loss = 0.0

    for inputs, labels in train_loader:

        inputs, labels = inputs.to(device), labels.to(device) # Move the data into device

        optimizer.zero_grad()

        labels -= 1

        outputs = model(inputs)

        loss = criterion(outputs, labels)



        loss.backward()

        optimizer.step()



        running_train_loss += loss.item()



    epoch_train_loss = running_train_loss / len(train_loader.dataset)



    model.eval()

    val_running_loss = 0.0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs, labels = inputs.to(device), labels.to(device) # Move the data into device

            outputs = model(inputs)

            labels -= 1

            loss = criterion(outputs, labels)

            val_running_loss += loss.item()



        epoch_val_loss = val_running_loss / len(val_loader.dataset)



    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')



    if epoch_val_loss < best_val_loss:

        print(f'Validation Loss Decreased({best_val_loss:.6f}--->{epoch_val_loss:.6f}) \t Saving The Model')

        best_val_loss = epoch_val_loss

        best_model_wts = copy.deepcopy(model.state_dict())

        no_improve_epochs = 0

    else:

        no_improve_epochs += 1



    if no_improve_epochs > patience:

        print('Early stopping!')

        model.load_state_dict(best_model_wts)

        break



end_time = time.time()  # Step 3: Record the end time

total_time = end_time - start_time  # Step 4: Calculate total training time



print(f'Finished training. Total training time: {total_time:.2f} seconds')  # Print the total training time
# ---- cell ----
# After the training loop, to record the final memory usage (GPU)

if torch.cuda.is_available():

    # Calculate the peak memory usage during the training

    peak_memory = torch.cuda.max_memory_allocated()

    # Calculate the final memory usage after training

    final_memory = torch.cuda.memory_allocated()



 # Calculate the memory used during the training

    memory_used = final_memory - initial_memory



    # Reset memory stats for future measurements

    torch.cuda.reset_peak_memory_stats()



    print(f"Peak Memory Allocated During Training: {peak_memory / 1e6} MB")

    print(f"Final Memory Allocated: {final_memory / 1e6} MB")

    # Adding the new line here to show memory used

    print(f"Memory Used During Training: {memory_used / 1e6} MB")
# ---- cell ----
# and it's been trained



replacement_value = m



# Dynamically construct the file name

model_save_path = 'Early_fusionp$num$_trento_model_state_dict.pth'

model_save_path = model_save_path.replace('$num$', str(replacement_value))



# Save the model state dictionary to the updated path

torch.save(model.state_dict(), model_save_path)
# ---- cell ----
# import torch

# import numpy as np

# from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

# import time  # Import the time module for timing the test phase



# Save the model

#model_save_path =  'Early_fusionp$num$_trento_model_state_dict.pth'

torch.save(model.state_dict(), model_save_path)

#replacement_value = m  # Replace "m" with this number dynamically



# Replace "$m$" with the integer value

#model_save_path = model_save_path.replace('$num$', str(replacement_value))





print(f'Model saved to {model_save_path}')



# Load the model (make sure to initialize the model architecture first)

model.load_state_dict(torch.load(model_save_path,weights_only=False))

model.to(device)



# Ensure the model is in evaluation mode

model.eval()



# Store predictions and actual labels

predictions = []

actual_labels = []



start_time = time.time()  # Start timing



with torch.no_grad():

    for hsi_patches, labels in test_loader:

        # Move data to the appropriate device

        hsi_patches = hsi_patches.to(device)

        labels -= 1  # Adjust labels if necessary



        # Forward pass

        outputs = model(hsi_patches)



        # Get predictions

        _, predicted = torch.max(outputs, 1)

        predictions.extend(predicted.cpu().numpy())

        actual_labels.extend(labels.cpu().numpy())



end_time = time.time()  # End timing

test_time = end_time - start_time  # Calculate the test time



# Optionally, calculate accuracy or other metrics using predictions and actual_labels



# Convert lists to NumPy arrays for easier manipulation

predictions_array = np.array(predictions)

actual_labels_array = np.array(actual_labels)



# Overall Accuracy

oa = accuracy_score(actual_labels_array, predictions_array)



# Confusion Matrix

cm = confusion_matrix(actual_labels_array, predictions_array)

# Calculate per-class accuracy from the confusion matrix

class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Average Accuracy

aa = np.mean(class_accuracy)



# Kappa Coefficient

kappa = cohen_kappa_score(actual_labels_array, predictions_array)



print(f'Overall Accuracy (OA): {oa:.4f}')

print(f'Average Accuracy (AA): {aa:.4f}')

print(f'Kappa Coefficient: {kappa:.4f}')

print(f'Test time: {test_time:.2f} seconds')  # Print the test time



for i, acc in enumerate(class_accuracy): print(f'Class {i+1} Accuracy: {acc:.4f}')
# ---- cell ----
#P1

Finished training. Total training time: 67.36 seconds

Model saved to Early_fusionp1_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9239

Average Accuracy (AA): 0.9324

Kappa Coefficient: 0.8986

Test time: 1.26 seconds

Class 1 Accuracy: 0.8428

Class 2 Accuracy: 0.9611

Class 3 Accuracy: 0.9786

Class 4 Accuracy: 0.9969

Class 5 Accuracy: 0.8741

Class 6 Accuracy: 0.9410
# ---- cell ----
#P3

Finished training. Total training time: 61.64 seconds

    Model saved to Early_fusionp3_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9878

Average Accuracy (AA): 0.9876

Kappa Coefficient: 0.9836

Test time: 1.78 seconds

Class 1 Accuracy: 0.9880

Class 2 Accuracy: 0.9741

Class 3 Accuracy: 1.0000

Class 4 Accuracy: 0.9994

Class 5 Accuracy: 0.9826

Class 6 Accuracy: 0.9817
# ---- cell ----
#P5

Finished training. Total training time: 69.50 seconds

Model saved to Early_fusionp5_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9949

Average Accuracy (AA): 0.9922

Kappa Coefficient: 0.9932

Test time: 1.89 seconds

Class 1 Accuracy: 0.9936

Class 2 Accuracy: 0.9712

Class 3 Accuracy: 1.0000

Class 4 Accuracy: 0.9999

Class 5 Accuracy: 0.9989

Class 6 Accuracy: 0.9895

# ---- cell ----
#P7

Finished training. Total training time: 93.78 seconds

    Model saved to Early_fusionp7_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9939

Average Accuracy (AA): 0.9904

Kappa Coefficient: 0.9919

Test time: 2.65 seconds

Class 1 Accuracy: 0.9895

Class 2 Accuracy: 0.9748

Class 3 Accuracy: 1.0000

Class 4 Accuracy: 1.0000

Class 5 Accuracy: 0.9999

Class 6 Accuracy: 0.9784
# ---- cell ----
#P9

Finished training. Total training time: 124.16 seconds

    Model saved to Early_fusionp9_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9962

Average Accuracy (AA): 0.9943

Kappa Coefficient: 0.9949

Test time: 3.53 seconds

Class 1 Accuracy: 0.9887

Class 2 Accuracy: 0.9874

Class 3 Accuracy: 1.0000

Class 4 Accuracy: 1.0000

Class 5 Accuracy: 0.9999

Class 6 Accuracy: 0.9898
# ---- cell ----
# patch=11 epoch=100

Finished training. Total training time: 165.42 seconds

Model saved to Early_fusionp11_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9972

Average Accuracy (AA): 0.9953

Kappa Coefficient: 0.9962

Test time: 4.51 seconds

Class 1 Accuracy: 0.9926

Class 2 Accuracy: 0.9921

Class 3 Accuracy: 0.9973

Class 4 Accuracy: 1.0000

Class 5 Accuracy: 1.0000

Class 6 Accuracy: 0.9898
# ---- cell ----
#P13

Finished training. Total training time: 217.60 seconds

Model saved to Early_fusionp13_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9968

Average Accuracy (AA): 0.9954

Kappa Coefficient: 0.9958

Test time: 5.84 seconds

Class 1 Accuracy: 0.9928

Class 2 Accuracy: 0.9946

Class 3 Accuracy: 1.0000

Class 4 Accuracy: 1.0000

Class 5 Accuracy: 0.9994

Class 6 Accuracy: 0.9856
# ---- cell ----
#P15

Finished training. Total training time: 250.93 seconds

    Model saved to Early_fusionp15_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9950

Average Accuracy (AA): 0.9918

Kappa Coefficient: 0.9933

Test time: 8.05 seconds

Class 1 Accuracy: 0.9936

Class 2 Accuracy: 0.9971

Class 3 Accuracy: 0.9973

Class 4 Accuracy: 1.0000

Class 5 Accuracy: 1.0000

Class 6 Accuracy: 0.9626
# ---- cell ----
#P17

Finished training. Total training time: 296.55 seconds

    Model saved to Early_fusionp17_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9958

Average Accuracy (AA): 0.9929

Kappa Coefficient: 0.9944

Test time: 9.18 seconds

Class 1 Accuracy: 0.9969

Class 2 Accuracy: 0.9827

Class 3 Accuracy: 0.9973

Class 4 Accuracy: 1.0000

Class 5 Accuracy: 0.9999

Class 6 Accuracy: 0.9803
# ---- cell ----
# patch=11 epoch=100 Normalisation 

Finished training. Total training time: 166.28 seconds

    Memory Used During Training: 0.246272 MB

Model saved to Early_fusionp11_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9972

Average Accuracy (AA): 0.9957

Kappa Coefficient: 0.9962

Test time: 4.52 seconds

Class 1 Accuracy: 0.9918

Class 2 Accuracy: 0.9910

Class 3 Accuracy: 1.0000

Class 4 Accuracy: 1.0000

Class 5 Accuracy: 1.0000

Class 6 Accuracy: 0.9915
# ---- cell ----
Finished training. Total training time: 124.35 seconds

Model saved to Early_fusionp1_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9378

Average Accuracy (AA): 0.9440

Kappa Coefficient: 0.9172

Test time: 1.65 seconds

Class 1 Accuracy: 0.8932

Class 2 Accuracy: 0.9741

Class 3 Accuracy: 0.9706

Class 4 Accuracy: 0.9965

Class 5 Accuracy: 0.8930

Class 6 Accuracy: 0.9368
# ---- cell ----
Finished training. Total training time: 100.57 seconds

Model saved to Early_fusionp3_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9890

Average Accuracy (AA): 0.9833

Kappa Coefficient: 0.9852

Test time: 1.68 seconds

Class 1 Accuracy: 0.9813

Class 2 Accuracy: 0.9615

Class 3 Accuracy: 0.9866

Class 4 Accuracy: 0.9978

Class 5 Accuracy: 0.9951

Class 6 Accuracy: 0.9777
# ---- cell ----
Finished training. Total training time: 49.65 seconds

Model saved to Early_fusionp5_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9902

Average Accuracy (AA): 0.9828

Kappa Coefficient: 0.9868

Test time: 1.73 seconds

Class 1 Accuracy: 0.9777

Class 2 Accuracy: 0.9536

Class 3 Accuracy: 0.9866

Class 4 Accuracy: 1.0000

Class 5 Accuracy: 0.9995

Class 6 Accuracy: 0.9794
# ---- cell ----
Finished training. Total training time: 76.70 seconds

Model saved to Early_fusionp7_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9904

Average Accuracy (AA): 0.9842

Kappa Coefficient: 0.9871

Test time: 1.82 seconds

Class 1 Accuracy: 0.9862

Class 2 Accuracy: 0.9683

Class 3 Accuracy: 0.9920

Class 4 Accuracy: 1.0000

Class 5 Accuracy: 0.9984

Class 6 Accuracy: 0.9604
# ---- cell ----
Finished training. Total training time: 53.48 seconds

Model saved to Early_fusionp9_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9902

Average Accuracy (AA): 0.9849

Kappa Coefficient: 0.9869

Test time: 2.08 seconds

Class 1 Accuracy: 0.9844

Class 2 Accuracy: 0.9662

Class 3 Accuracy: 1.0000

Class 4 Accuracy: 1.0000

Class 5 Accuracy: 0.9989

Class 6 Accuracy: 0.9600
# ---- cell ----
Finished training. Total training time: 68.32 seconds

Model saved to Early_fusionp11_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9932

Average Accuracy (AA): 0.9872

Kappa Coefficient: 0.9909

Test time: 2.28 seconds

Class 1 Accuracy: 0.9944

Class 2 Accuracy: 0.9816

Class 3 Accuracy: 0.9866

Class 4 Accuracy: 1.0000

Class 5 Accuracy: 0.9999

Class 6 Accuracy: 0.9607
# ---- cell ----
Finished training. Total training time: 80.24 seconds

Model saved to Early_fusionp13_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9905

Average Accuracy (AA): 0.9852

Kappa Coefficient: 0.9872

Test time: 2.46 seconds

Class 1 Accuracy: 0.9992

Class 2 Accuracy: 0.9845

Class 3 Accuracy: 1.0000

Class 4 Accuracy: 1.0000

Class 5 Accuracy: 0.9983

Class 6 Accuracy: 0.9292
# ---- cell ----
Finished training. Total training time: 152.45 seconds

Model saved to Early_fusionp1_trento_model_state_dict.pth

Overall Accuracy (OA): 0.8520

Average Accuracy (AA): 0.8663

Kappa Coefficient: 0.8032

Test time: 1.71 seconds

Class 1 Accuracy: 0.8796

Class 2 Accuracy: 0.8078

Class 3 Accuracy: 0.9920

Class 4 Accuracy: 0.8794

Class 5 Accuracy: 0.8407

Class 6 Accuracy: 0.7982
# ---- cell ----
Finished training. Total training time: 155.26 seconds

Model saved to Early_fusionp13_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9719

Average Accuracy (AA): 0.9615

Kappa Coefficient: 0.9623

Test time: 2.40 seconds

Class 1 Accuracy: 0.9949

Class 2 Accuracy: 0.8834

Class 3 Accuracy: 1.0000

Class 4 Accuracy: 0.9767

Class 5 Accuracy: 0.9984

Class 6 Accuracy: 0.9158
# ---- cell ----
Finished training. Total training time: 105.63 seconds

Model saved to Early_fusionp11_Uh2013_model_state_dict.pth

Overall Accuracy (OA): 0.9751

Average Accuracy (AA): 0.9594

Kappa Coefficient: 0.9666

Test time: 2.27 seconds

Class 1 Accuracy: 0.9918

Class 2 Accuracy: 0.8949

Class 3 Accuracy: 0.9786

Class 4 Accuracy: 0.9935

Class 5 Accuracy: 0.9960

Class 6 Accuracy: 0.9014
# ---- cell ----
Finished training. Total training time: 135.61 seconds

Model saved to Early_fusionp9_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9701

Average Accuracy (AA): 0.9580

Kappa Coefficient: 0.9600

Test time: 2.11 seconds

Class 1 Accuracy: 0.9867

Class 2 Accuracy: 0.9039

Class 3 Accuracy: 0.9920

Class 4 Accuracy: 0.9988

Class 5 Accuracy: 0.9810

Class 6 Accuracy: 0.8856
# ---- cell ----
Finished training. Total training time: 95.08 seconds

Model saved to Early_fusionp7_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9701

Average Accuracy (AA): 0.9577

Kappa Coefficient: 0.9599

Test time: 1.81 seconds

Class 1 Accuracy: 0.9931

Class 2 Accuracy: 0.8970

Class 3 Accuracy: 0.9947

Class 4 Accuracy: 0.9916

Class 5 Accuracy: 0.9873

Class 6 Accuracy: 0.8827
# ---- cell ----
Finished training. Total training time: 136.45 seconds

Model saved to Early_fusionp5_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9620

Average Accuracy (AA): 0.9534

Kappa Coefficient: 0.9492

Test time: 1.73 seconds

Class 1 Accuracy: 0.9949

Class 2 Accuracy: 0.9010

Class 3 Accuracy: 0.9893

Class 4 Accuracy: 0.9831

Class 5 Accuracy: 0.9704

Class 6 Accuracy: 0.8817
# ---- cell ----
Finished training. Total training time: 156.99 seconds

Model saved to Early_fusionp3_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9163

Average Accuracy (AA): 0.9095

Kappa Coefficient: 0.8873

Test time: 1.67 seconds

Class 1 Accuracy: 0.9357

Class 2 Accuracy: 0.8909

Class 3 Accuracy: 0.9893

Class 4 Accuracy: 0.8675

Class 5 Accuracy: 0.9968

Class 6 Accuracy: 0.7769
# ---- cell ----
# Before the training loop, to record the initial memory usage (GPU)

if torch.cuda.is_available():

    torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats at the start

    initial_memory = torch.cuda.memory_allocated()

    print(f"Initial Memory Allocated: {initial_memory / 1e6} MB")
# ---- cell ----
# # 2.0 HSI+Lidar

# import torch

# import torch.nn as nn

# import torch.optim as optim

# from torch.utils.data import DataLoader

# import copy

# import time  # Step 1: Import the time module





# Instantiate the model

model = HSILidClassificationMambaModel(

    spatial_dim=m,

    hsi_num_bands=63,

    lidar_num_bands=1,

    hidden_dim=1024,

    output_dim=512,

    delta_param_init=0.1,

    num_classes=6

)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.000001)



epochs = 200

best_val_loss = float('inf')

best_model_wts = copy.deepcopy(model.state_dict())

patience = 50



start_time = time.time()  # Step 2: Record the start time

# Updated Training Loop with HSI and LiDAR data

for epoch in range(epochs):

    model.train()  # Set model to training mode

    running_train_loss = 0.0



    # Iterate over data.

    for (hsi_inputs, labels), (lidar_inputs, _) in zip(hsi_train_loader, Lidar_train_loader):

        hsi_inputs = hsi_inputs.to(device)

        lidar_inputs = lidar_inputs.to(device)

        labels -= 1

        labels = labels.to(device)



        optimizer.zero_grad()  # Zero the parameter gradients





        # Forward pass

        outputs = model(hsi_inputs, lidar_inputs)

        loss = criterion(outputs, labels)



        # Backward pass and optimize

        loss.backward()

        optimizer.step()



        running_train_loss += loss.item() * hsi_inputs.size(0)



    epoch_train_loss = running_train_loss / len(hsi_train_loader.dataset)



    # Validation phase

    model.eval()

    val_running_loss = 0.0

    with torch.no_grad():

        for (hsi_inputs, labels), (lidar_inputs, _) in zip(hsi_val_loader, Lidar_val_loader):

            hsi_inputs = hsi_inputs.to(device)

            lidar_inputs = lidar_inputs.to(device)

            labels -= 1

            labels = labels.to(device)



            outputs = model(hsi_inputs, lidar_inputs)

            loss = criterion(outputs, labels)



            val_running_loss += loss.item() * hsi_inputs.size(0)



        epoch_val_loss = val_running_loss / len(hsi_val_loader.dataset)



    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')



    # Check if this is the best model (based on validation loss)

    if epoch_val_loss < best_val_loss:

        print(f'Validation Loss Decreased({best_val_loss:.6f}--->{epoch_val_loss:.6f}) \t Saving The Model')

        best_val_loss = epoch_val_loss

        best_model_wts = copy.deepcopy(model.state_dict())

        no_improve_epochs = 0

    else:

        no_improve_epochs += 1



    # Early stopping

    if no_improve_epochs > patience:

        print('Early stopping!')

        break



model.load_state_dict(best_model_wts)  # Load best model weights



end_time = time.time()  # Record the end time

total_time = end_time - start_time  # Calculate total training time

print(f'Finished training. Total training time: {total_time:.2f} seconds')  #
# ---- cell ----
# After the training loop, to record the final memory usage (GPU)

if torch.cuda.is_available():

    # Calculate the peak memory usage during the training

    peak_memory = torch.cuda.max_memory_allocated()

    # Calculate the final memory usage after training

    final_memory = torch.cuda.memory_allocated()



    # Calculate the memory used during the training

    memory_used = final_memory - initial_memory



    # Reset memory stats for future measurements

    torch.cuda.reset_peak_memory_stats()



    print(f"Peak Memory Allocated During Training: {peak_memory / 1e6} MB")

    print(f"Final Memory Allocated: {final_memory / 1e6} MB")

    # Adding the new line here to show memory used

    print(f"Memory Used During Training: {memory_used / 1e6} MB")# After the training loop, to record the final memory usage (GPU)

# ---- cell ----


replacement = m



# Dynamically construct the file name

model_save_path = 'mid_fusionp$num$_trento_model_state_dict.pth'

model_save_path = model_save_path.replace('$num$', str(replacement))



# Save the model state dictionary to the updated path

torch.save(model.state_dict(), model_save_path)
# ---- cell ----
# import torch

# import numpy as np

# from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

# import time



#model_save_path =  'mid_fusionp$num$_trento_model_state_dict.pth'

torch.save(model.state_dict(), model_save_path)

#replacement_value = m  # Replace "m" with this number dynamically



# Replace "$m$" with the integer value

#model_save_path = model_save_path.replace('$num$', str(replacement_value))

print(f'Model saved to {model_save_path}')

# Load the model

model.load_state_dict(torch.load(model_save_path,weights_only=False))

model.to(device)

model.eval()  # Set the model to evaluation mode



predictions = []

actual_labels = []



start_time = time.time()



with torch.no_grad():

    for (hsi_patches, hsi_labels), (lidar_patches, _) in zip(hsi_test_loader, Lidar_test_loader):

        hsi_patches = hsi_patches.to(device)

        lidar_patches = lidar_patches.to(device)

        hsi_labels -= 1

        hsi_labels = hsi_labels.to(device)



        # Forward pass using both HSI and LiDAR patches

        outputs = model(hsi_patches, lidar_patches)



        # Get predictions

        _, predicted = torch.max(outputs, 1)

        predictions.extend(predicted.cpu().numpy())

        actual_labels.extend(hsi_labels.cpu().numpy())



end_time = time.time()

test_time = end_time - start_time



# Convert lists to NumPy arrays for calculation

predictions_array = np.array(predictions)

actual_labels_array = np.array(actual_labels)



# Calculate metrics

oa = accuracy_score(actual_labels_array, predictions_array)

cm = confusion_matrix(actual_labels_array, predictions_array)

class_accuracy = cm.diagonal() / cm.sum(axis=1)

aa = np.mean(class_accuracy)

kappa = cohen_kappa_score(actual_labels_array, predictions_array)



print(f'Overall Accuracy (OA): {oa:.4f}')

print(f'Average Accuracy (AA): {aa:.4f}')

print(f'Kappa Coefficient: {kappa:.4f}')

print(f'Test time: {test_time:.2f} seconds')



for i, acc in enumerate(class_accuracy): print(f'Class {i+1} Accuracy: {acc:.4f}')
# ---- cell ----
Finished training. Total training time: 213.02 seconds

Model saved to mid_fusionp1_trento_model_state_dict.pth

Overall Accuracy (OA): 0.8557

Average Accuracy (AA): 0.8688

Kappa Coefficient: 0.8079

Test time: 2.38 seconds

Class 1 Accuracy: 0.8873

Class 2 Accuracy: 0.8143

Class 3 Accuracy: 0.9866

Class 4 Accuracy: 0.8961

Class 5 Accuracy: 0.8330

Class 6 Accuracy: 0.7952
# ---- cell ----
Finished training. Total training time: 142.33 seconds

Model saved to mid_fusionp13_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9783

Average Accuracy (AA): 0.9680

Kappa Coefficient: 0.9709

Test time: 3.28 seconds

Class 1 Accuracy: 0.9839

Class 2 Accuracy: 0.9172

Class 3 Accuracy: 0.9973

Class 4 Accuracy: 0.9957

Class 5 Accuracy: 0.9944

Class 6 Accuracy: 0.9194
# ---- cell ----
Finished training. Total training time: 91.29 seconds

Model saved to mid_fusionp11_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9733

Average Accuracy (AA): 0.9631

Kappa Coefficient: 0.9642

Test time: 3.08 seconds

Class 1 Accuracy: 0.9752

Class 2 Accuracy: 0.9172

Class 3 Accuracy: 0.9866

Class 4 Accuracy: 0.9787

Class 5 Accuracy: 0.9973

Class 6 Accuracy: 0.9233
# ---- cell ----


Finished training. Total training time: 209.29 seconds

Model saved to Early_fusionp9_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9734

Average Accuracy (AA): 0.9623

Kappa Coefficient: 0.9643

Test time: 2.87 seconds

Class 1 Accuracy: 0.9859

Class 2 Accuracy: 0.8920

Class 3 Accuracy: 0.9947

Class 4 Accuracy: 0.9812

Class 5 Accuracy: 0.9983

Class 6 Accuracy: 0.9220
# ---- cell ----
Finished training. Total training time: 107.16 seconds

Model saved to mid_fusionp7_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9710

Average Accuracy (AA): 0.9607

Kappa Coefficient: 0.9612

Test time: 2.62 seconds

Class 1 Accuracy: 0.9844

Class 2 Accuracy: 0.8787

Class 3 Accuracy: 1.0000

Class 4 Accuracy: 0.9816

Class 5 Accuracy: 0.9941

Class 6 Accuracy: 0.9256
# ---- cell ----
Finished training. Total training time: 118.57 seconds

Model saved to mid_fusionp5_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9636

Average Accuracy (AA): 0.9500

Kappa Coefficient: 0.9512

Test time: 2.52 seconds

Class 1 Accuracy: 0.9836

Class 2 Accuracy: 0.8589

Class 3 Accuracy: 0.9866

Class 4 Accuracy: 0.9777

Class 5 Accuracy: 0.9887

Class 6 Accuracy: 0.9047
# ---- cell ----
Finished training. Total training time: 217.86 seconds

Model saved to mid_fusionp3_trento_model_state_dict.pth

Overall Accuracy (OA): 0.9502

Average Accuracy (AA): 0.9472

Kappa Coefficient: 0.9333

Test time: 2.43 seconds

Class 1 Accuracy: 0.9898

Class 2 Accuracy: 0.8549

Class 3 Accuracy: 0.9947

Class 4 Accuracy: 0.9702

Class 5 Accuracy: 0.9495

Class 6 Accuracy: 0.9243
# ---- cell ----
# ---- cell ----
# ---- cell ----
# ---- cell ----
# ---- cell ----
# ---- cell ----
# ---- cell ----