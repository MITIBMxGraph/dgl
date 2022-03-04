#ifndef FAST_SAMPLER_CONFIG_H_
#define FAST_SAMPLER_CONFIG_H_


#include "fs_common.hpp"


class FastSamplerConfig : public dgl::runtime::Object {
  public:
    FastSamplerConfig( 
      size_t batch_size,
      dgl::NDArray x,
      optional<dgl::NDArray> y,
      dgl::NDArray rowptr,
      dgl::NDArray col,
      dgl::NDArray idx,
      std::vector<int64_t> sizes,
      bool skip_nonfull_batch,
      bool pin_memory
    );

    size_t batch_size;
    dgl::NDArray x;
    optional<dgl::NDArray> y;
    dgl::NDArray rowptr;
    dgl::NDArray col;
    dgl::NDArray idx;
    std::vector<int64_t> sizes;
    bool skip_nonfull_batch;
    bool pin_memory;

    static constexpr const char* _type_key = "salient.FastSamplerConfig";
    DGL_DECLARE_OBJECT_TYPE_INFO(FastSamplerConfig, dgl::runtime::Object);
};


DGL_DEFINE_OBJECT_REF(FastSamplerConfigRef, FastSamplerConfig);


#endif // FAST_SAMPLER_CONFIG_H_
