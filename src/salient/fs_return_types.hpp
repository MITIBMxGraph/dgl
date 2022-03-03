#ifndef FAST_SAMPLER_RETURN_TYPES_H_
#define FAST_SAMPLER_RETURN_TYPES_H_


#include "fs_common.hpp"


/*
 * Object which is an array of dgl::HeteroGraphRef.
 * Array is implemented with std::vector.
 */
class HeteroGraphArray : public dgl::runtime::Object {
  public:
    explicit HeteroGraphArray(std::vector<dgl::HeteroGraphRef> graphs): graphs(graphs) {}
    HeteroGraphArray(){}
    virtual ~HeteroGraphArray() = default;
    std::vector<dgl::HeteroGraphRef> graphs;

    static constexpr const char* _type_key = "graph.HeteroGraphArray";
    DGL_DECLARE_OBJECT_TYPE_INFO(HeteroGraphArray, dgl::runtime::Object);
};

DGL_DEFINE_OBJECT_REF(HeteroGraphArrayRef, HeteroGraphArray);


/*
 * Output after multilayer sampling.
 * 1. tensor of indices to slice
 * 2. array of message flow graphs
 */
typedef struct ProtoSample {
  torch::Tensor indices;
  HeteroGraphArrayRef mfgs;
};


/*
 * Output after multilayer sampling and slicing.
 * 1. tensor of sliced featuers
 * 2. optionally, tensor of sliced labels
 * 3. array of message flow graphs
 * 4. corresponding range of indices trained on (post shuffle)
 */
typedef struct PreparedSample {
  dgl::NDArray x;
  optional<dgl::NDArray> y;
  HeteroGraphArrayRef mfgs;
  std::pair<int32_t, int32_t> range;
};


class OptionalPreparedSample : public dgl::runtime::Object {
  public:
    OptionalPreparedSample(){}
    virtual ~OptionalPreparedSample() = default;
    optional<PreparedSample> value; 

    static constexpr const char* _type_key = "salient.OptionalPreparedSample";
    DGL_DECLARE_OBJECT_TYPE_INFO(OptionalPreparedSample, dgl::runtime::Object);
};

DGL_DEFINE_OBJECT_REF(OptionalPreparedSampleRef, OptionalPreparedSample);


#endif // FAST_SAMPLER_RETURN_TYPES_H_
