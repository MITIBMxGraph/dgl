#include "fs_config.hpp"


FastSamplerConfig::FastSamplerConfig(
  size_t batch_size_,
  dgl::NDArray x_,
  optional<dgl::NDArray> y_,
  dgl::NDArray rowptr_,
  dgl::NDArray col_,
  dgl::NDArray idx_,
  std::vector<int64_t> sizes_,
  bool skip_nonfull_batch_,
  bool pin_memory_)
  : batch_size{batch_size_}, x{x_}, y{y_}, rowptr{rowptr_},
    col{col_}, idx{idx_}, sizes{sizes_}, skip_nonfull_batch{skip_nonfull_batch_},
    pin_memory{pin_memory_} {}


/* Methods */

DGL_REGISTER_GLOBAL("salient._CAPI_FSConfigCreate")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {

  size_t batch_size = args[0];
  dgl::NDArray x = args[1];
  optional<dgl::NDArray> y = args[2];
  dgl::NDArray rowptr = args[3];
  dgl::NDArray col = args[4];
  dgl::NDArray idx = args[5];
  std::vector<int64_t> sizes = args[6];
  bool skip_nonfull_batch = args[7];
  bool pin_memory = args[8];

  auto fsc = std::make_shared<FastSamplerConfig>();
  // debugging, switch to a constructor
  fsc->batch_size = batch_size;
  fsc->x = x;
  fsc->y = y;
  fsc->rowptr = rowptr;
  fsc->col = col;
  fsc->idx = idx;
  fsc->sizes = sizes;
  fsc->skip_nonfull_batch = skip_nonfull_batch;
  fsc->pin_memory = pin_memory;

  *rv = FastSamplerConfigRef(fsc);

});

DGL_REGISTER_GLOBAL("salient._CAPI_FSConfigGetBatchSize")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
  FastSamplerConfigRef fscr = args[0];
  *rv = static_cast<int>(fscr->batch_size);
});

DGL_REGISTER_GLOBAL("salient._CAPI_FSConfigIsPinMemory")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
  FastSamplerConfigRef fscr = args[0];
  *rv = (fscr->pin_memory) ? 1 : 0;
});

DGL_REGISTER_GLOBAL("salient._CAPI_FSConfigSkipNonfullBatch")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
  FastSamplerConfigRef fscr = args[0];
  *rv = (fscr->skip_nonfull_batch) ? 1 : 0;
});

DGL_REGISTER_GLOBAL("salient._CAPI_FSConfigGetNumTrainIdxs")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
  FastSamplerConfigRef fscr = args[0];
  *rv = fscr->idx.NumElements();
});
