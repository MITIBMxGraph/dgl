from dgl._ffi.function import _init_api
from dgl._ffi.object import register_object, ObjectBase
from dgl.heterograph import DGLBlock

_init_api("dgl.salient.mfg")

def get_cpp_test_mfg():
    return DGLBlock(_CAPI_TestMFG())

def get_cpp_test_mfg_vector():
    print(_CAPI_TestMFGVector())
    return [DGLBlock(g) for g in _CAPI_TestMFGVector()]


# class that contains multiple dgl graph objects created in the C++ backend
@register_object('graph.HeteroGraphArray')
class HeteroGraphIndexArray(ObjectBase):
    # define getitem magic method to access underlying C++ vector 
    def __getitem__(self, idx):
        return _CAPI_DGLHeteroArrayGetGraphAtIdx(self, idx)
