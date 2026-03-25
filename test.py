import torch

meta = torch.load('/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset/metadata.pt', weights_only=False)
print("jumbo_y_idx:", meta['jumbo_y_idx'])
print("jumbo_x_idx:", meta['jumbo_x_idx'])
print("input_shape:", meta['input_shape'])