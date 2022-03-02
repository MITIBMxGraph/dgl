from dgl._ffi.function import _init_api
from dgl._ffi.object import register_object, ObjectBase
from dgl.heterograph import DGLBlock


_init_api("dgl.salient")


@register_object('graph.HeteroGraphArray')
class HeteroGraphIndexArray(ObjectBase):
    """ An array of dgl::HeteroGraphRef objects. """

    def __getitem__(self, idx):
        return _CAPI_DGLHeteroArrayGetGraphAtIdx(self, idx)

    def __len__(self):
        return _CAPI_DGLHeteroArrayGetLen(self)


@register_object('salient.OptionalPreparedSample')
class OptionalPreparedSample(ObjectBase):
    """ An optional type containing:
        - batch features
        - batch labels (optional)
        - batch mfgs
        - range of indices trained on in batch
    """
    pass


"""
Test functions.
"""


def get_cpp_test_mfg():
    return DGLBlock(_CAPI_TestMFG())

def get_cpp_test_mfg_vector():
    print(_CAPI_TestMFGVector())
    return [DGLBlock(g) for g in _CAPI_TestMFGVector()]

def test_none_rv(x):
    return _CAPI_TestNone(x)
