from ._ffi.function import _init_api

def add(a, b):
    return MyAdd(a, b)

_init_api("dgl.calculator")
