#ifndef FAST_SAMPLER_SAMPLE_AND_SLICE_H_
#define FAST_SAMPLER_SAMPLE_AND_SLICE_H_


#include "fs_common.hpp"
#include "fs_thread.hpp"
#include "fs_session.hpp"
#include "sample_cpu.hpp"


ProtoSample multilayer_sample(
    std::vector<int64_t> n_ids,
    std::vector<int64_t> const& sizes,
    torch::Tensor rowptr,
    torch::Tensor col,
    bool pin_memory = false
); 

ProtoSample multilayer_sample(
    torch::Tensor idx,
    std::vector<int64_t> const& sizes,
    torch::Tensor rowptr,
    torch::Tensor col,
    bool pin_memory = false
);

template <typename scalar_t>
dgl::NDArray serial_index_impl(
    torch::Tensor const in,
    torch::Tensor const idx,
    int64_t const n,
    bool const pin_memory = false
);

template <typename scalar_t>
dgl::NDArray serial_index_impl(
    torch::Tensor const in,
    torch::Tensor const idx,
    bool const pin_memory = false
);

dgl::NDArray serial_index(
    torch::Tensor const in,
    torch::Tensor const idx,
    int64_t const n,
    bool const pin_memory = false
);

dgl::NDArray serial_index(
    torch::Tensor const in,
    torch::Tensor const idx,
    bool const pin_memory = false
);


#endif // FAST_SAMPLER_SAMPLE_AND_SLICE_H_
