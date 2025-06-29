from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort
from memory import UnsafePointer
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
import math
from complex import ComplexFloat32
from utils.index import IndexList
from gpu.host import DeviceContext
from gpu import thread_idx, block_dim, block_idx

@export
fn PyInit_gobrr() -> PythonObject:
    try:
        var m = PythonModuleBuilder("gobrr")
        m.def_function[project_3d_to_2d_cpu]("project_3d_to_2d_cpu")
        m.def_function[project_3d_to_2d_gpu]("project_3d_to_2d_gpu")
        return m.finalize()
    except e:
        return abort[PythonObject](String("Error creating Mojo module: ", e))


alias N_PROJECTIONS = 2500
alias VOLUME_DFT_DEPTH = 256
alias VOLUME_DFT_HEIGHT = 256
alias VOLUME_DFT_WIDTH = 129
alias OUTPUT_DFT_HEIGHT = VOLUME_DFT_HEIGHT
alias OUTPUT_DFT_WIDTH = VOLUME_DFT_WIDTH


alias VolumeDFTLayout = Layout.row_major(2, VOLUME_DFT_DEPTH, VOLUME_DFT_HEIGHT, VOLUME_DFT_WIDTH)
alias VolumeDFT = LayoutTensor[
    DType.float32,
    VolumeDFTLayout
]

alias RotationMatricesLayout = Layout.row_major(N_PROJECTIONS, 3, 3)
alias RotationMatrices = LayoutTensor[
    DType.float32,
    RotationMatricesLayout
]

alias OutputDFTsLayout = Layout.row_major(N_PROJECTIONS, 2, 256, 129)
alias OutputDFTs = LayoutTensor[
    DType.float32,
    OutputDFTsLayout
]


def project_3d_to_2d_cpu(
    volume_dft: PythonObject,
    rotation_matrices: PythonObject,
    output_dfts: PythonObject
):
    var volume_dft_ptr = volume_dft.data_ptr().unsafe_get_as_pointer[DType.float32]()
    var rotation_matrices_ptr = rotation_matrices.data_ptr().unsafe_get_as_pointer[DType.float32]()
    var output_dfts_ptr = output_dfts.data_ptr().unsafe_get_as_pointer[DType.float32]()
    mojo_projection_cpu(volume_dft_ptr, rotation_matrices_ptr, output_dfts_ptr)


def project_3d_to_2d_gpu(
    volume_dft: PythonObject,
    rotation_matrices: PythonObject,
    output_dfts: PythonObject
):
    # grab pointers to host memory
    var volume_dft_host_ptr = volume_dft.data_ptr().unsafe_get_as_pointer[DType.float32]()
    var rotation_matrices_host_ptr = rotation_matrices.data_ptr().unsafe_get_as_pointer[DType.float32]()
    var output_dfts_host_ptr = output_dfts.data_ptr().unsafe_get_as_pointer[DType.float32]()

    # construct cpu layout tensors around input/output data pointers
    # this is so we can get the sizes to allocate on the gpu
    var volume_dft_host = LayoutTensor[DType.float32, VolumeDFTLayout](volume_dft_host_ptr)
    var rotation_matrices_host = LayoutTensor[DType.float32, RotationMatricesLayout](rotation_matrices_host_ptr)
    var output_dfts_host = LayoutTensor[DType.float32, OutputDFTsLayout](output_dfts_host_ptr)

    # create gpu context
    with DeviceContext() as ctx:
        # allocate buffers on device
        volume_dft_device = ctx.enqueue_create_buffer[DType.float32](volume_dft_host.size())
        rotation_matrices_device = ctx.enqueue_create_buffer[DType.float32](rotation_matrices_host.size())
        output_dfts_device = ctx.enqueue_create_buffer[DType.float32](output_dfts_host.size())

        # copy data from host to device
        ctx.enqueue_copy(volume_dft_device, volume_dft_host_ptr)
        ctx.enqueue_copy(rotation_matrices_device, rotation_matrices_host_ptr)

        # actually launch the kernel
        ctx.enqueue_function[mojo_projection_gpu](
            volume_dft_device.unsafe_ptr(),
            rotation_matrices_device.unsafe_ptr(),
            output_dfts_device.unsafe_ptr(),
            grid_dim=(2500, 32, 17),
            block_dim=(1, 8, 8)
        )

        # copy results back to host
        ctx.enqueue_copy(output_dfts_host_ptr, output_dfts_device)

        # wait for everything to be done
        ctx.synchronize()

fn mojo_projection_cpu(
    volume_dft_ptr: UnsafePointer[Float32],
    rotation_matrices_ptr: UnsafePointer[Float32],
    output_dfts_ptr: UnsafePointer[Float32]
):
    # construct layout tensors around input/output data pointers
    var volume_dft = LayoutTensor[DType.float32, VolumeDFTLayout](volume_dft_ptr)
    var rotation_matrices = LayoutTensor[DType.float32, RotationMatricesLayout](rotation_matrices_ptr)
    var output_dfts = LayoutTensor[DType.float32, OutputDFTsLayout](output_dfts_ptr)

    # for loops: loop over each dimension of the output dfts
    for imagei in range(N_PROJECTIONS):
        for y in range(OUTPUT_DFT_HEIGHT):
            for x in range(OUTPUT_DFT_WIDTH):

                # construct sample point, in 3d?
                var sx: Float32 = Float32(x)*0.5/128.0
                var step_size_y: Float32 = 0.5 / 128.0
                var y_dim_length: Float32 = 1.0 - step_size_y
                var sy: Float32 = (y_dim_length * (Float32(y) / 255.0)) - 0.5
                var sz: Float32 = 0.0

                # filter sample points based on magnitude
                var mag2: Float32 = sx*sx + sy*sy
                if mag2 > 0.25:
                    continue
                    # TODO: we still need to write a pixel here?

                # rotate sample point
                # couldn't type the function inputs correctly so inlining
                #var (sx_rot, sy_rot, sz_rot) = rotate((sx, sy, sz), mojo_rotation_matrices, imagei)
                var r00 = rotation_matrices[imagei, 0, 0]
                var r01 = rotation_matrices[imagei, 0, 1]
                var r02 = rotation_matrices[imagei, 0, 2]
                var r10 = rotation_matrices[imagei, 1, 0]
                var r11 = rotation_matrices[imagei, 1, 1]
                var r12 = rotation_matrices[imagei, 1, 2]
                var r20 = rotation_matrices[imagei, 2, 0]
                var r21 = rotation_matrices[imagei, 2, 1]
                var r22 = rotation_matrices[imagei, 2, 2]

                var xrot = r00 * sx + r01 * sy + r02 * sz
                var yrot = r10 * sx + r11 * sy + r12 * sz
                var zrot = r20 * sx + r21 * sy + r22 * sz

                # if sampled point from "bad space" (x < 0), remember that we need to conjugate
                var needs_conjugation = xrot < 0.0

                # transform into tensor coordinate space
                # y and z have same dim lengths and dc positions because volume is cubic
                xrot = xrot * 2.0 * 128
                var dc_y = Float32(256 // 2)
                yrot = ((yrot + 0.5) / y_dim_length) * 255.0
                zrot = ((zrot + 0.5) / y_dim_length) * 255.0

                # if we're conjugating, invert the sample point through the origin
                if needs_conjugation:
                    xrot *= -1.0
                    yrot *= -1.0
                    zrot *= -1.0

                # discretize sample along x
                var x0f = math.floor(xrot)
                if x0f < 0.0:
                    continue
                    # TODO: what pixel do we write in this case?
                var tx = xrot - x0f
                var x0 = Int(x0f)
                var x1 = x0 + 1
                if x1 > 128:
                    continue
                    # TODO: don't forget to write a pixel

                # discretize sample along y
                var y0f = math.floor(yrot)
                if y0f < 0.0:
                    continue
                    # TODO: write pixel
                var ty = yrot - y0f
                var y0 = Int(y0f)
                var y1 = y0 + 1
                if y1 > 255:
                    continue
                    # TODO: write pixel

                # discretize along z
                var z0f = math.floor(zrot)
                if z0f < 0.0:
                    continue
                    # TODO: write pixel
                var tz = zrot - z0f
                var z0 = Int(z0f)
                var z1 = z0 + 1
                if z1 > 255:
                    continue
                    # TODO: write pixel

                # sample the DFT
                var re: Float32 = lerp_3d(
                    volume_dft,
                    0,
                    x0, x1, tx[0],
                    y0, y1, ty[0],
                    z0, z1, tz[0]
                )
                var im: Float32 = lerp_3d(
                    volume_dft,
                    1,
                    x0, x1, tx[0],
                    y0, y1, ty[0],
                    z0, z1, tz[0]
                )

                # if we're conjugating, do that
                if needs_conjugation:
                    im *= -1.0

                # write answer
                output_dfts[imagei, 0, y, x][0] = re
                output_dfts[imagei, 1, y, x][0] = im


fn mojo_projection_gpu(
    volume_dft_ptr: UnsafePointer[Float32],
    rotation_matrices_ptr: UnsafePointer[Float32],
    output_dfts_ptr: UnsafePointer[Float32]
):
    # construct layout tensors around input/output data pointers
    var volume_dft = LayoutTensor[DType.float32, VolumeDFTLayout](volume_dft_ptr)
    var rotation_matrices = LayoutTensor[DType.float32, RotationMatricesLayout](rotation_matrices_ptr)
    var output_dfts = LayoutTensor[DType.float32, OutputDFTsLayout](output_dfts_ptr)

    # recreate imagei, y and x from thread indices
    var imagei = block_idx.z
    var y = block_dim.y * block_idx.y + thread_idx.y
    var x = block_dim.x * block_idx.x + thread_idx.x

    # remember to guard x because we have overflow lol
    if x > (OUTPUT_DFT_WIDTH - 1):
        return

    # construct sample point, in 3d?
    var sx: Float32 = Float32(x)*0.5/128.0
    var step_size_y: Float32 = 0.5 / 128.0
    var y_dim_length: Float32 = 1.0 - step_size_y
    var sy: Float32 = (y_dim_length * (Float32(y) / 255.0)) - 0.5
    var sz: Float32 = 0.0

    # filter sample points based on magnitude
    var mag2: Float32 = sx*sx + sy*sy
    if mag2 > 0.25:
        output_dfts[imagei, 0, y, x][0] = 0.0
        output_dfts[imagei, 1, y, x][0] = 0.0
        return

    # rotate sample point
    # couldn't type the function inputs correctly so inlining
    #var (sx_rot, sy_rot, sz_rot) = rotate((sx, sy, sz), mojo_rotation_matrices, imagei)
    var r00 = rotation_matrices[imagei, 0, 0]
    var r01 = rotation_matrices[imagei, 0, 1]
    var r02 = rotation_matrices[imagei, 0, 2]
    var r10 = rotation_matrices[imagei, 1, 0]
    var r11 = rotation_matrices[imagei, 1, 1]
    var r12 = rotation_matrices[imagei, 1, 2]
    var r20 = rotation_matrices[imagei, 2, 0]
    var r21 = rotation_matrices[imagei, 2, 1]
    var r22 = rotation_matrices[imagei, 2, 2]

    var xrot = r00 * sx + r01 * sy + r02 * sz
    var yrot = r10 * sx + r11 * sy + r12 * sz
    var zrot = r20 * sx + r21 * sy + r22 * sz

    # if sampled point from "bad space" (x < 0), remember that we need to conjugate
    var needs_conjugation = xrot < 0.0

    # transform into tensor coordinate space
    # y and z have same dim lengths and dc positions because volume is cubic
    xrot = xrot * 2.0 * 128
    var dc_y = Float32(256 // 2)
    yrot = ((yrot + 0.5) / y_dim_length) * 255.0
    zrot = ((zrot + 0.5) / y_dim_length) * 255.0

    # if we're conjugating, invert the sample point through the origin
    if needs_conjugation:
        xrot *= -1.0
        yrot *= -1.0
        zrot *= -1.0

    # discretize sample along x
    var x0f = math.floor(xrot)
    if x0f < 0.0:
        output_dfts[imagei, 0, y, x][0] = 0.0
        output_dfts[imagei, 1, y, x][0] = 0.0
        return

    var tx = xrot - x0f
    var x0 = Int(x0f)
    var x1 = x0 + 1
    if x1 > OUTPUT_DFT_WIDTH - 1:
        output_dfts[imagei, 0, y, x][0] = 0.0
        output_dfts[imagei, 1, y, x][0] = 0.0
        return

    # discretize sample along y
    var y0f = math.floor(yrot)
    if y0f < 0.0:
        output_dfts[imagei, 0, y, x][0] = 0.0
        output_dfts[imagei, 1, y, x][0] = 0.0
        return

    var ty = yrot - y0f
    var y0 = Int(y0f)
    var y1 = y0 + 1
    if y1 > OUTPUT_DFT_HEIGHT - 1:
        output_dfts[imagei, 0, y, x][0] = 0.0
        output_dfts[imagei, 1, y, x][0] = 0.0
        return

    # discretize along z
    var z0f = math.floor(zrot)
    if z0f < 0.0:
        output_dfts[imagei, 0, y, x][0] = 0.0
        output_dfts[imagei, 1, y, x][0] = 0.0
        return

    var tz = zrot - z0f
    var z0 = Int(z0f)
    var z1 = z0 + 1
    if z1 > OUTPUT_DFT_HEIGHT - 1:
        output_dfts[imagei, 0, y, x][0] = 0.0
        output_dfts[imagei, 1, y, x][0] = 0.0
        return

    # sample the DFT
    var re: Float32 = lerp_3d(
        volume_dft,
        0,
        x0, x1, tx[0],
        y0, y1, ty[0],
        z0, z1, tz[0]
    )
    var im: Float32 = lerp_3d(
        volume_dft,
        1,
        x0, x1, tx[0],
        y0, y1, ty[0],
        z0, z1, tz[0]
    )

    # if we're conjugating, do that
    if needs_conjugation:
        im *= -1.0

    # write answer
    output_dfts[imagei, 0, y, x][0] = re
    output_dfts[imagei, 1, y, x][0] = im





fn lerp_1d(v0: Float32, v1: Float32, t: Float32) -> Float32:
    return (t*v1) + (1.0 - t)*v0

fn lerp_3d(
    volume_dft: VolumeDFT,
    ci: Int,
    x0: Int,
    x1: Int,
    tx: Float32,
    y0: Int,
    y1: Int,
    ty: Float32,
    z0: Int,
    z1: Int,
    tz: Float32
) -> Float32:

    var lerp_z0y0: Float32 = lerp_1d(
        volume_dft[ci, z0, y0, x0][0],
        volume_dft[ci, z0, y0, x1][0],
        tx
    )
    var lerp_z0y1: Float32 = lerp_1d(
        volume_dft[ci, z0, y1, x0][0],
        volume_dft[ci, z0, y1, x1][0],
        tx
    )

    var lerp_z1y0: Float32 = lerp_1d(
        volume_dft[ci, z1, y0, x0][0],
        volume_dft[ci, z1, y0, x1][0],
        tx
    )
    var lerp_z1y1: Float32 = lerp_1d(
        volume_dft[ci, z1, y1, x0][0],
        volume_dft[ci, z1, y1, x1][0],
        tx
    )

    var lerp_z0: Float32 = lerp_1d(lerp_z0y0, lerp_z0y1, ty)
    var lerp_z1: Float32 = lerp_1d(lerp_z1y0, lerp_z1y1, ty)

    return lerp_1d(lerp_z0, lerp_z1, tz)

