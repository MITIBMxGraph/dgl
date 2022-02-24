#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>

DGL_REGISTER_GLOBAL("salient.multiplier.MyMul")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    int a = args[0];
    int c = args[1];
    *rv = a * c;
  });
