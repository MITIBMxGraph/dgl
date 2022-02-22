import os
import sys
from pathlib import Path
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension
from torch.__config__ import parallel_info


def flags_to_list(flagstring):
    return list(filter(bool, flagstring.split(' ')))


WITH_SYMBOLS = True if os.getenv('WITH_SYMBOLS', '0') == '1' else False
CXX_FLAGS = flags_to_list(os.getenv('CXX_FLAGS', ''))
ROOT_PATH = Path(__file__).resolve().parent

def get_extensions():
    define_macros = []
    libraries = []
    library_dirs = []
    #libraries = ['libdgl']
    #library_dirs = ['/home/gridsan/pmurzynowski/dgl/build/']
    extra_compile_args = {'cxx': ['-O3', '-mcpu=native', '-std=c++17', '-ggdb3', '-Wall'] + CXX_FLAGS}
    # '-lnvToolsExt' to link to nvtx api annotation capability
    #extra_link_args = ['-lnvToolsExt'] if WITH_SYMBOLS else ['-lnvToolsExt', '-s']
    extra_link_args = ['-lnvToolsExt']
    dgl_link_args = [
            #'-L/home/gridsan/pmurzynowski/dgl/build/third_party/dmlc-core/libdmlc.a',
            #'-L/home/gridsan/pmurzynowski/dgl/build/tensoradapter/pytorch/libtensoradapter_pytorch_1.10.1.so',
            #'-L/home/gridsan/pmurzynowski/dgl/python/dgl/_ffi/_cy3/core.cpython-39-x86_64-linux-gnu.so',
            #'-L/home/gridsan/pmurzynowski/dgl/build/libdgl.so'
            #'-L/home/gridsan/pmurzynowski/dgl/build --no-as-needed -llibdgl'
            #'-L/home/gridsan/pmurzynowski/dgl/build -ldgl -Wl,--no-as-needed'
            #'-L/home/gridsan/pmurzynowski/dgl/build -ldgl' 
            '-ldgl'
    ]
    extra_link_args += dgl_link_args

    info = parallel_info()
    if 'backend: OpenMP' in info and 'OpenMP not found' not in info:
        extra_compile_args['cxx'] += ['-DAT_PARALLEL_OPENMP']
        if sys.platform == 'win32':
            extra_compile_args['cxx'] += ['/openmp']
        else:
            extra_compile_args['cxx'] += ['-fopenmp']
    else:
        print('Compiling without OpenMP...')

    include_dirs = [ROOT_PATH, ROOT_PATH.joinpath('parallel-hashmap'),
    # hacked-in dgl include
        '/home/gridsan/pmurzynowski/dgl/third_party/dlpack/include',
        '/home/gridsan/pmurzynowski/dgl/third_party/dmlc-core/include',
        #'/home/gridsan/pmurzynowski/dgl/third_party/METIS/GKlib',
        #'/home/gridsan/pmurzynowski/dgl/third_party/METIS/include/',
        '/home/gridsan/pmurzynowski/dgl/include/',
        #'/home/gridsan/pmurzynowski/dgl/src/'
    ]

    """
    # get all dgl cc files, hacky
    dgl_src_files = []
    for r, d, f in os.walk('/home/gridsan/pmurzynowski/dgl/src/'):
        for file in f:
            if '.cc' in file:
                dgl_src_files.append(os.path.join(r, file))
    """

    return [
        CppExtension(
            'fast_sampler',
            ['fast_sampler.cpp'],
            #['fast_sampler.cpp'],
            #['fast_sampler.cpp'] + dgl_src_files,
            #[
            #    'fast_sampler.cpp',
            #    '/home/gridsan/pmurzynowski/dgl/src/array/array.cc',
            #    '/home/gridsan/pmurzynowski/dgl/src/runtime/c_object_api.cc',
            #    '/home/gridsan/pmurzynowski/dgl/src/runtime/object.cc'
            #],
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=libraries,
            library_dirs=library_dirs,
        ),
    ]

setup(
    name='fast_sampler',
    ext_modules=get_extensions(),
    package_data={'':['/home/gridsan/pmurzynowski/dgl/build/libdgl.so']},
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    })
