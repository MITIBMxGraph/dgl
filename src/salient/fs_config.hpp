#ifndef FAST_SAMPLER_CONFIG_H_
#define FAST_SAMPLER_CONFIG_H_


#include "fs_common.hpp"
#include "fs_types.hpp"


class FastSamplerConfig : public dgl::runtime::Object {
  public:
    /*
    FastSamplerConfig( 
      size_t batch_size,
      dgl::NDArray x,
      OptionalNDArrayRef y,
      dgl::NDArray rowptr,
      dgl::NDArray col,
      dgl::NDArray idx,
      dgl::NDArray sizes,
      bool skip_nonfull_batch,
      bool pin_memory
    );
    */
    FastSamplerConfig(){}

    size_t batch_size;
    dgl::NDArray x;
    OptionalNDArrayRef y;
    dgl::NDArray rowptr;
    dgl::NDArray col;
    dgl::NDArray idx;
    dgl::NDArray sizes;
    bool skip_nonfull_batch;
    bool pin_memory;

    static constexpr const char* _type_key = "salient.FastSamplerConfig";
    DGL_DECLARE_OBJECT_TYPE_INFO(FastSamplerConfig, dgl::runtime::Object);
};


DGL_DEFINE_OBJECT_REF(FastSamplerConfigRef, FastSamplerConfig);


#endif // FAST_SAMPLER_CONFIG_H_
