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

# allocate stuff and pass to mojo
import einops
dft_real = torch.view_as_real(dft) # (d, h, w, 2)
dft_real = einops.rearrange(dft_real, "d h w c -> c d h w")
dft_real = dft_real.contiguous() # (2, d, h, w)
c, d, h, w = dft_real.shape
out_dfts = torch.zeros(size=(N_PROJECTIONS_PER_BATCH, c, h, w), dtype=torch.float32, device=dft_real.device)


gobrr.project_3d_to_2d_gpu(dft_real, rotation_matrices, out_dfts)
print("done yo")
out_dfts = einops.rearrange(out_dfts, "b c h w -> b h w c")
out_dfts = torch.view_as_complex(out_dfts.contiguous())
print(out_dfts.shape, out_dfts.dtype)
projections = torch.fft.ifftshift(out_dfts, dim=(-2,))  # ifftshift of 2D rfft
projections = torch.fft.irfftn(projections, dim=(-2, -1))
projections = torch.fft.ifftshift(projections, dim=(-2, -1))  # recenter 2D image in real space

# unpad if required
projections = projections[..., pad_length:-pad_length, pad_length:-pad_length]
projections = torch.real(projections)
mrcfile.write("gpu.mrc", projections.numpy(), overwrite=True)

# # warmup block for python
# for i in range(N_WARMUP_ITERATIONS):
#     tilt_series_dft = extract_central_slices_rfft_3d(
#         volume_rfft=dft,
#         image_shape=volume.shape,
#         rotation_matrices=torch.tensor(rotation_matrices).float(),
#     )  # (..., h, w) rfft stack

# # actual perf measurement for python
# print("python start")
# start = datetime.now()
# for i in range(N_ITERATIONS):
#     tilt_series_dft = extract_central_slices_rfft_3d(
#         volume_rfft=dft,
#         image_shape=volume.shape,
#         rotation_matrices=rotation_matrices,
#     )  # (..., h, w) rfft stack
# end = datetime.now()
# print(f"pps python: {(N_PROJECTIONS_PER_BATCH * N_ITERATIONS) / (end - start).total_seconds():.2f}")

# time.sleep(3)
# # warmup block for mojo
# print("mojo start")
# for i in range(N_WARMUP_ITERATIONS):
#     tilt_series_dft = gobrr.project_3d_to_2d_cpu(dft_real, rotation_matrices, out_dfts)

# # actual perf measurement for mojo
# start = datetime.now()
# for i in range(N_ITERATIONS):
#     tilt_series_dft = gobrr.project_3d_to_2d_cpu(dft_real, rotation_matrices, out_dfts)
# end = datetime.now()
# print(f"pps mojo cpu: {(N_PROJECTIONS_PER_BATCH * N_ITERATIONS) / (end - start).total_seconds():.2f}")


# start = datetime.now()
# tilt_series_dft = mojo_blabla()
# end = datetime.now()
#
# is_close_enough = torch.allclose(tilt_series_dft, tilt_series_dft, atol=1e-5)
# print(f"{is_close_enough=}")
#
# # transform back to real space
# projections = torch.fft.ifftshift(tilt_series_dft, dim=(-2,))  # ifftshift of 2D rfft
# projections = torch.fft.irfftn(projections, dim=(-2, -1))
# projections = torch.fft.ifftshift(projections, dim=(-2, -1))  # recenter 2D image in real space
#
# # unpad if required
# projections = projections[..., pad_length:-pad_length, pad_length:-pad_length]
# projections = torch.real(projections)
#
#
# reconstruction = backproject_2d_to_3d(
#     images=projections,
#     rotation_matrices=torch.tensor(rotation_matrices).float(),
# )
#
# reconstruction_dft = torch.fft.fftn(reconstruction, dim=(-3, -2, -1))
# reconstruction_dft = torch.fft.fftshift(reconstruction_dft, dim=(-3, -2, -1))
# reconstruction_ps = torch.log(torch.real(torch.abs(reconstruction_dft)) ** 2)
#
# # import napari
# #
# # viewer = napari.Viewer()
# # viewer.add_image(volume.numpy())
# # # viewer.add_image(projections.numpy())
# # viewer.add_image(reconstruction.numpy())
# # viewer.add_image(reconstruction_ps.numpy())
# # napari.run()

# # Viz logic for mojo output
# # transform back to real space
# out_dfts = einops.rearrange(out_dfts, "b c h w -> b h w c")
# out_dfts = torch.view_as_complex(out_dfts.contiguous())
# print(out_dfts.shape, out_dfts.dtype)
# projections = torch.fft.ifftshift(out_dfts, dim=(-2,))  # ifftshift of 2D rfft
# projections = torch.fft.irfftn(projections, dim=(-2, -1))
# projections = torch.fft.ifftshift(projections, dim=(-2, -1))  # recenter 2D image in real space
#
# # unpad if required
# projections = projections[..., pad_length:-pad_length, pad_length:-pad_length]
# projections = torch.real(projections)
#
# import napari
# viewer = napari.Viewer()
# viewer.add_image(projections.numpy(), name="projections")
# napari.run()