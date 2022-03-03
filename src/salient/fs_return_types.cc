#include "fs_return_types.hpp"


/* HeteroGraphArray methods. */


DGL_REGISTER_GLOBAL("salient._CAPI_DGLHeteroArrayGetGraphAtIdx")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    HeteroGraphArrayRef hgar = args[0];
    int idx = args[1];
    *rv = hgar->graphs[idx];
});

DGL_REGISTER_GLOBAL("salient._CAPI_DGLHeteroArrayGetLen")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    HeteroGraphArrayRef hgar = args[0];
    *rv = static_cast<int>(hgar->graphs.size());
});


/* OptionalPreparedSample methods. 
 *
 * From dgl docs:
 *  The reflection is indeed a little bit slow due to the string key lookup.
 *  To speed it up, you could define an attribute access API
 * Though less pretty the access API approach is used.
 * Dereferencing to access optional values is not too pretty as well.
 */


DGL_REGISTER_GLOBAL("salient._CAPI_OptionalPreparedSampleHasValue")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    OptionalPreparedSampleRef opsr = args[0];
    int has = (opsr->value) ? 1 : 0;
    *rv = has;
});

DGL_REGISTER_GLOBAL("salient._CAPI_OptionalPreparedSampleHasY")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    OptionalPreparedSampleRef opsr = args[0];
    int has = (opsr->value->y) ? 1 : 0;
    *rv = has;
});

DGL_REGISTER_GLOBAL("salient._CAPI_OptionalPreparedSampleGetMfgs")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    OptionalPreparedSampleRef opsr = args[0];
    *rv = opsr->value->mfgs;
});

DGL_REGISTER_GLOBAL("salient._CAPI_OptionalPreparedSampleGetRangeStart")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    OptionalPreparedSampleRef opsr = args[0];
    *rv = opsr->value->range.first;
});

DGL_REGISTER_GLOBAL("salient._CAPI_OptionalPreparedSampleGetRangeEnd")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    OptionalPreparedSampleRef opsr = args[0];
    *rv = opsr->value->range.second;
});

DGL_REGISTER_GLOBAL("salient._CAPI_OptionalPreparedSampleGetX")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    OptionalPreparedSampleRef opsr = args[0];
    *rv = opsr->value->x;
});

DGL_REGISTER_GLOBAL("salient._CAPI_OptionalPreparedSampleGetY")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    OptionalPreparedSampleRef opsr = args[0];
    *rv = *(opsr->value->y);
});
