import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from phantominator import shepp_logan

from DcTNN.dc import fft_2d, ifft_2d
from DcTNN.model import TokenVIT, axVIT, cascadeNet
from train_utils import FastMRIMaskGenerator

norm = "ortho"
N = 320
R = 4
numCh = 1
lamb = True

# Generate phantom for testing
ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
ph = torch.tensor(ph.copy(), dtype=torch.float32).unsqueeze(0)
ph = torch.cat([ph] * numCh, dim=0).unsqueeze(0)

# Undersample with the fastMRI mask generator
mask_generator = FastMRIMaskGenerator([R])
y_full = fft_2d(ph)
y, sampling_mask, _ = mask_generator.apply(y_full, R, seed=0)
zf_image = ifft_2d(y, norm=norm)[:, :numCh, :, :]

# Define transformer encoder parameters
patchSize = 16
nhead_patch = 8
nhead_axial = 8
layerNo = 1
d_model_axial = None
d_model_patch = None
num_encoder_layers = 2
dim_feedforward = None

patchArgs = {
    "patch_size": patchSize,
    "tokenizer_type": "patch",
    "layerNo": layerNo,
    "numCh": numCh,
    "nhead": nhead_patch,
    "num_encoder_layers": num_encoder_layers,
    "dim_feedforward": dim_feedforward,
    "d_model": d_model_patch,
}
kdArgs = {
    "patch_size": patchSize,
    "tokenizer_type": "kaleidoscope",
    "layerNo": layerNo,
    "numCh": numCh,
    "nhead": nhead_patch,
    "num_encoder_layers": num_encoder_layers,
    "dim_feedforward": dim_feedforward,
    "d_model": d_model_patch,
}
axArgs = {
    "layerNo": layerNo,
    "numCh": numCh,
    "d_model": d_model_axial,
    "nhead": nhead_axial,
    "num_encoder_layers": num_encoder_layers,
    "dim_feedforward": dim_feedforward,
}

encList = [axVIT, TokenVIT, TokenVIT]
encArgs = [axArgs, kdArgs, patchArgs]

dcenc = cascadeNet(N, encList, encArgs, lamb, learning="image")

pytorch_total_params = sum(p.numel() for p in dcenc.parameters() if p.requires_grad)
print("Number of trainable params:", pytorch_total_params)

with torch.no_grad():
    start = time.time()
    phRecon = dcenc(zf_image, y, sampling_mask)
    elapsed = time.time() - start
    print("Execution took", elapsed, "seconds")

plt.close()
plt.figure(1)
plt.imshow(np.abs(ph[0, 0].cpu()))
plt.title("Phantom")

plt.figure(2)
plt.imshow(np.abs(zf_image[0, 0].cpu()))
plt.title("Zero Fill Image")

plt.figure(3)
plt.imshow(np.abs(phRecon[0, 0].cpu()))
plt.title("Reconstruction")

plt.figure(4)
plt.imshow(sampling_mask.squeeze().cpu())
plt.title("Sampling Mask")
plt.show()
