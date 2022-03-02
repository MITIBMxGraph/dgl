#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>

DGL_REGISTER_GLOBAL("calculator.MyAdd")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    int a = args[0];
    int b = args[1];
    *rv = a + b;
  });
