# Lyrebird: 3D Conformer Generation

This repository contains the **Lyrebird** neural network model for 3D conformer generation, developed by Rowan using flow matching techniques. You can use the pretrained model weights locally or run predictions directly via the [Rowan web platform](https://labs.rowansci.com/).

For questions or issues, please open a GitHub issue or contact the Rowan team at contact@rowansci.com.

## Overview

Lyrebird generates high-quality 3D molecular conformers from SMILES strings using a flow-based generative model. The model is trained on diverse molecular geometries and can generate multiple conformers for conformational analysis, molecular docking, and other computational chemistry applications.

## Example Usage

```python
import torch
from lyrebird import LyrebirdCalculator

# Initialize the calculator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
calculator = LyrebirdCalculator("lyrebird.yaml", "lyrebird.ckpt", device=device)

# Generate conformers from SMILES
smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # ibuprofen
conformers = calculator.predict(smiles, num_initial_confs=5)

print(f"Generated {len(conformers)} conformers")
print(f"First conformer shape: {conformers[0].shape}")  # (N_atoms, 3)
```

## Local Usage

Install the required packages using:
```bash
conda env create -f environment.yml
conda activate lyrebird-env
```

To run the example script:
```bash
python example.py
```

This model can run on either CPU or GPU. GPU is recommended for faster generation of multiple conformers.

## License

This model is released under the MIT License. See [LICENSE](LICENSE) for details.
