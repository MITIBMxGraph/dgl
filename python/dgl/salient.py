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

    def __bool__(self):
        return True if _CAPI_OptionalPreparedSampleHasValue(self) else False

    @property
    def x(self):
       return _CAPI_OptionalPreparedSampleGetX(self) if _CAPI_OptionalPreparedSampleHasValue(self) else None

    @property
    def y(self):
       return _CAPI_OptionalPreparedSampleGetY(self) if (_CAPI_OptionalPreparedSampleHasValue(self) and _CAPI_OptionalPreparedSampleHasY(self)) else None

    @property
    def blocks(self):
        if _CAPI_OptionalPreparedSampleHasValue(self):
            mfgs = _CAPI_OptionalPreparedSampleGetMfgs(self)
            return [DGLBlock(mfgs[i]) for i in range(len(mfgs))]
        return None

    @property
    def range(self):
       return (_CAPI_OptionalPreparedSampleGetRangeStart(self), _CAPI_OptionalPreparedSampleGetRangeEnd(self)) if _CAPI_OptionalPreparedSampleHasValue(self) else None


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
