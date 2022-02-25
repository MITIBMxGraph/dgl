from dgl._ffi.function import _init_api
from dgl.heterograph import DGLBlock

_init_api("dgl.salient.mfg")

def get_cpp_test_mfg():
    return DGLBlock(_CAPI_TestMFG())

