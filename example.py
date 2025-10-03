import torch
import yaml
from etflow import BaseFlow

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open("lyrebird.yaml", "r") as f:
    config = yaml.safe_load(f)

model = BaseFlow.from_config(config)
checkpoint = torch.load("lyrebird.ckpt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()

# Generate conformers for ibuprofen
ibuprofen_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
conformers = model.predict([ibuprofen_smiles], num_samples=5, device=device)

print(f"Generated {len(conformers[ibuprofen_smiles])} conformers")
print(f"First conformer shape: {list(conformers[ibuprofen_smiles])[0].shape}")
