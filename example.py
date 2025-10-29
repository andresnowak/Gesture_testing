import torch

from src.stgcn import STGCN


if __name__ == "__main__":
    # --- Hyperparameters ---
    batch_size = 8
    num_nodes = 5         # A small graph with 5 nodes
    num_features = 1      # Each node has 1 feature (e.g., traffic speed)
    num_timesteps = 12    # We look at 12 previous time steps
    out_features = 3      # We want to predict 3 future time steps

    # --- Create a dummy adjacency matrix for a 5-node chain graph (0-1-2-3-4) ---
    A = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0

    print("Original Adjacency Matrix (A):\n", A)

    # --- Correctly Normalize the Adjacency Matrix ---
    # This is a critical step for Graph Convolutional Networks
    A_tilde = A + torch.eye(num_nodes)
    D_tilde = torch.diag(torch.sum(A_tilde, dim=1))
    D_tilde_inv_sqrt = torch.pow(D_tilde, -0.5)
    D_tilde_inv_sqrt[torch.isinf(D_tilde_inv_sqrt)] = 0. # Handle cases of isolated nodes

    A_hat = torch.mm(torch.mm(D_tilde_inv_sqrt, A_tilde), D_tilde_inv_sqrt)
    print("\nNormalized Adjacency Matrix (A_hat):\n", A_hat)

    # --- Instantiate the Model ---
    model = STGCN(
        num_nodes=num_nodes,
        num_features=num_features,
        num_timesteps=num_timesteps,
        out_features=out_features
    )

    # --- Create a random input tensor ---
    # Shape: (Batch, Time, Nodes, Features)
    x = torch.randn(batch_size, num_timesteps, num_nodes, num_features)

    # --- Perform a forward pass ---
    output = model(x, A_hat)

    # --- Print shapes to verify ---
    print("\n--- Shape Verification ---")
    print("Input shape (x):", x.shape)
    print("Adjacency matrix shape (A_hat):", A_hat.shape)
    print("Output shape:", output.shape)
    print("Expected output shape:", (batch_size, num_nodes, out_features))

    # --- Check if the output shape is correct ---
    assert output.shape == (batch_size, num_nodes, out_features)
    print("\nSuccess! The output shape is correct.")
