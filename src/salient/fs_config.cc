#include "fs_config.hpp"


FastSamplerConfig::FastSamplerConfig(
  size_t batch_size_,
  torch::Tensor x_,
  optional<torch::Tensor> y_,
  torch::Tensor rowptr_,
  torch::Tensor col_,
  torch::Tensor idx_,
  std::vector<int64_t> sizes_,
  bool skip_nonfull_batch_,
  bool pin_memory_)
  : batch_size{batch_size_}, x{x_}, y{y_}, rowptr{rowptr_},
    col{col_}, idx{idx_}, sizes{sizes_}, skip_nonfull_batch{skip_nonfull_batch_},
    pin_memory{pin_memory_} {}


/* Methods */


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
    *rv = fscr->idx.numel();
});
