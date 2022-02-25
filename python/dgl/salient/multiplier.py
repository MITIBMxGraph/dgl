from dgl._ffi.function import _init_api

_init_api("dgl.salient.multiplier")

def mul(a, b):
    return _CAPI_MyMul(a, b)

