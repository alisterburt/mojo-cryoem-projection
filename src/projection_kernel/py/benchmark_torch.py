import time
from datetime import datetime

import mrcfile
import torch
import torch.nn.functional as F
import numpy as np
from torch_fourier_slice import project_3d_to_2d, backproject_2d_to_3d, extract_central_slices_rfft_3d
from scipy.spatial.transform import Rotation as R
from torch_grid_utils import fftfreq_grid

# benchmark parameters
N_PROJECTIONS_PER_BATCH = 2500
N_WARMUP_ITERATIONS = 3
N_ITERATIONS = 20

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

# send to cuda
dft = dft.to(dtype=torch.float32, device=torch.device("cuda"))
rotation_matrices = torch.as_tensor(rotation_matrices, device=dft.device, dtype=torch.float32)

# warmup block for python
for i in range(N_WARMUP_ITERATIONS):
    tilt_series_dft = extract_central_slices_rfft_3d(
        volume_rfft=dft,
        image_shape=volume.shape,
        rotation_matrices=rotation_matrices,
    )  # (..., h, w) rfft stack

# actual perf measurement for python
start = datetime.now()
for i in range(N_ITERATIONS):
    out_dfts = extract_central_slices_rfft_3d(
        volume_rfft=dft,
        image_shape=volume.shape,
        rotation_matrices=rotation_matrices,
    )  # (..., h, w) rfft stack
end = datetime.now()
print(f"pps torch (cuda): {(N_PROJECTIONS_PER_BATCH * N_ITERATIONS) / (end - start).total_seconds():.2f}")

# convert to real space and write out projections
projections = torch.fft.ifftshift(out_dfts, dim=(-2,))  # ifftshift of 2D rfft
projections = torch.fft.irfftn(projections, dim=(-2, -1))
projections = torch.fft.ifftshift(projections, dim=(-2, -1))  # recenter 2D image in real space
projections = projections[..., pad_length:-pad_length, pad_length:-pad_length]
projections = torch.real(projections)
mrcfile.write("torch_output.mrc", projections.numpy(), overwrite=True)