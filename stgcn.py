import math

import torch
from torch import nn
import torch.nn.functional as F


class TimeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        # We use Conv2d is used as the input we will expect is of 4D, so using kernel (1, kernel_size) makes it behave as 1D

        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels, 2 * out_channels, (1, kernel_size)
        )  # 2 * out_channels as we will separate the results, and this makes it just one fused operation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform one graph-convolution step using the normalized Laplacian.

        Args:
            x (torch.Tensor): Node feature matrix of shape (Batch, F_in, N, Time),
                                where N is number of nodes and F_in is feature size.
        Returns:
            torch.Tensor: Updated node features of shape (Batch, F_out, N, Time_f).
        """

        out = self.conv(x)  # (batch, 2 * out_channels, N, time_f)
        p_out = out[:, 0 : self.out_channels]  # (batch, out_channels, N, time_f)
        q_out = out[:, self.out_channels :]  # (batch, out_channels, N, time_f)

        x = p_out * F.sigmoid(q_out)  # gate

        return x


class STGCNBlock(nn.Module):
    def __init__(
        self, in_channels: int, spatial_channels: int, out_channels: int, num_nodes: int
    ):
        super().__init__()

        self.temp_block_1 = TimeBlock(
            in_channels=in_channels, out_channels=out_channels
        )

        # Here spatial just means that we go back to a spatial dimension
        self.Theta = nn.Parameter(
            torch.FloatTensor(out_channels, spatial_channels)
        )  # The filter. (k, out_channels, spatial_channels), and here hops k=1

        self.temp_block_2 = TimeBlock(
            in_channels=spatial_channels, out_channels=out_channels
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Theta)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        Perform one graph-convolution step using the normalized Laplacian.

        Args:
            x (torch.Tensor): Node feature matrix of shape (Batch, F_in, N, Time),
                                where N is number of nodes and F_in is feature size.
            A_hat (torch.Tensor): Normalized Laplacian (or adjacency) matrix
                                        of the graph, shape (N, N).

        Returns:
            torch.Tensor: Updated node features of shape (Batch, F_out, N, time).
        """

        temp_conv = self.temp_block_1(x)  # (batch, out_channel, N, time)
        temp_conv = F.relu(temp_conv)

        # 1st order approximation Graph convolution
        # Filter theta applies a frequency response g(λ) to the signal’s spectral components at those eigenvalues (so it acts on the components of signal x along each eigenvector of Laplacian)

        # Channel mix (Θ): (B, out_channel, N, time) @ (Cout, Cspat) -> (B, Cspat, N, time)
        h = torch.einsum("bfnt,fs->bsnt", temp_conv, self.Theta)
        # Node mix (Â): (N, N) @ (B, Cspat, N, Time) -> (B, Cspat, N, Time)
        y = torch.einsum("ij,bsjt->bsit", A_hat, h)
        conv = F.relu(y)

        # temp block again
        temp_conv = self.temp_block_2(conv)
        x = F.relu(temp_conv)  # (batch, out_channels, N, time)

        return x


class STGCN(nn.Module):
    def __init__(
        self, num_nodes: int, num_features: int, num_timesteps: int, out_features: int
    ):
        super().__init__()

        self.block_1 = STGCNBlock(
            in_channels=num_features,
            out_channels=64,
            spatial_channels=16,
            num_nodes=num_nodes,
        )
        self.block_2 = STGCNBlock(
            in_channels=64, out_channels=64, spatial_channels=16, num_nodes=num_nodes
        )
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)

        self.out = nn.Linear(
            (num_timesteps - 2 * 5) * 64, out_features
        )  # because the time dimension gets reduced with the kernel_size = 3

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        Perform one graph-convolution step using the normalized Laplacian.

        Args:
            x (torch.Tensor): Node feature matrix of shape (Batch, Time, N, F_in),
                                where N is number of nodes and F_in is feature size.
            A_hat (torch.Tensor): Normalized Laplacian (or adjacency) matrix
                                        of the graph, shape (N, N).

        Returns:
            torch.Tensor: Updated node features of shape (Batch, F_out, N, time).
        """
        x = x.permute(0, 3, 2, 1)  # (batch, in_features, N, time)

        st_out = self.block_1(x, A_hat)  # (batch, out_channel, N, time)
        st_out = self.block_2(st_out, A_hat)  # (batch, out_channel, N, time)

        x = self.last_temporal(st_out)  # (batch, out_channel, N, time)
        x = F.relu(x)

        x = x.permute(0, 2, 1, 3) # (batch, N, out_channel, time_f)
        x = x.reshape((x.shape[0], x.shape[1], -1)) # (batch, N, out_channel * time_f)

        x = self.out(x)

        return x
