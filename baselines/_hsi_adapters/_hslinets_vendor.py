"""Model classes extracted verbatim from the vendored HSLiNets notebook.

Source: vendor/HSLiNets-original/Model/HSLiNets_FusionMamba_Trento_*.ipynb
(commit ee315e991bd6a48ed5941a093cd9f73a62020645).
Only the four class cells below were copied; nothing was edited.  The notebook
does not contain the dual-modal (HSI+LiDAR) class -- see
baselines/hslinets/README.md for the documented adaptation choice.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
