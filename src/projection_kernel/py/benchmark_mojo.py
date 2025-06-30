import time
from datetime import datetime

import mrcfile
import torch
import torch.nn.functional as F
import numpy as np
from torch_fourier_slice import project_3d_to_2d, backproject_2d_to_3d, extract_central_slices_rfft_3d
from scipy.spatial.transform import Rotation as R
from torch_grid_utils import fftfreq_grid

# make mojo stuff available
import sys
sys.path.append(".")
import gobrr

# benchmark parameters
N_PROJECTIONS_PER_BATCH = 2500
N_WARMUP_ITERATIONS = 3
N_ITERATIONS = 1

# benchmarking in python
volume = mrcfile.read("data/4v6x_box128_4apx.mrc")
volume = torch.as_tensor(volume, dtype=torch.float32)

tilt_angles = np.linspace(-60, 60, num=N_PROJECTIONS_PER_BATCH)
rotation_matrices = R.from_euler("y", angles=tilt_angles, degrees=True).as_matrix()
rotation_matrices = torch.tensor(rotation_matrices, dtype=torch.float32)

## Data prep for projection (project 3D down to 2D)

# padding
pad_length = volume.shape[-1] // 2
volume = F.pad(volume, pad=[pad_length] * 6, mode='constant', value=0)

# premultiply by sinc2
grid = fftfreq_grid(
    image_shape=volume.shape,
    rfft=False,
    fftshift=True,
    norm=True,
    device=volume.device
)
volume = volume * torch.sinc(grid) ** 2

# calculate DFT
dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))  # volume center to array origin
dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
dft = torch.fft.fftshift(dft, dim=(-3, -2,))  # actual fftshift of 3D rfft

# prepare dft for sending to mojo
# (need to send over as float32 not complex)
import einops
dft_real = torch.view_as_real(dft) # (d, h, w, 2)
dft_real = einops.rearrange(dft_real, "d h w c -> c d h w")
dft_real = dft_real.contiguous() # (2, d, h, w)
c, d, h, w = dft_real.shape
out_dfts = torch.zeros(size=(N_PROJECTIONS_PER_BATCH, c, h, w), dtype=torch.float32, device=dft_real.device)

# benchmark mojo gpu code
gobrr.benchmark_project_3d_to_2d_gpu(dft_real, rotation_matrices, out_dfts)

# convert to real space and write out projections
out_dfts = einops.rearrange(out_dfts, "b c h w -> b h w c")
out_dfts = torch.view_as_complex(out_dfts.contiguous())
projections = torch.fft.ifftshift(out_dfts, dim=(-2,))  # ifftshift of 2D rfft
projections = torch.fft.irfftn(projections, dim=(-2, -1))
projections = torch.fft.ifftshift(projections, dim=(-2, -1))  # recenter 2D image in real space
projections = projections[..., pad_length:-pad_length, pad_length:-pad_length]
projections = torch.real(projections)
mrcfile.write("mojo_output.mrc", projections.cpu().numpy(), overwrite=True)
