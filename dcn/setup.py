from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc_ARCH  = ['-arch=sm_52']
nvcc_ARCH += ["-gencode=arch=compute_61,code=\"compute_61\""]
# nvcc_ARCH += ["-gencode=arch=compute_75,code=\"sm_75\""]
# nvcc_ARCH += ["-gencode=arch=compute_70,code=\"sm_70\""]
nvcc_ARCH += ["-gencode=arch=compute_61,code=\"sm_61\""]
nvcc_ARCH += ["-gencode=arch=compute_52,code=\"sm_52\""]
extra_compile_args = { 
            'cxx': ['-Wno-unused-function', '-Wno-write-strings'],
            'nvcc': nvcc_ARCH,}

setup(
    name='deform_conv',
    ext_modules=[
        CUDAExtension('deform_conv_cuda', [
            'src/deform_conv_cuda.cpp',
            'src/deform_conv_cuda_kernel.cu',
        ],
        extra_compile_args=extra_compile_args,
        ),
        CUDAExtension('deform_pool_cuda', [
            'src/deform_pool_cuda.cpp', 'src/deform_pool_cuda_kernel.cu'
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
