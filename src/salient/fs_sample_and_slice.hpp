#ifndef FAST_SAMPLER_SAMPLE_AND_SLICE_H_
#define FAST_SAMPLER_SAMPLE_AND_SLICE_H_


#include "fs_common.hpp"
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
torch::Tensor serial_index_impl(
    torch::Tensor const in,
    torch::Tensor const idx,
    int64_t const n,
    bool const pin_memory = false
);

template <typename scalar_t>
torch::Tensor serial_index_impl(
    torch::Tensor const in,
    torch::Tensor const idx,
    bool const pin_memory = false
);

torch::Tensor serial_index(
    torch::Tensor const in,
    torch::Tensor const idx,
    int64_t const n,
    bool const pin_memory = false
);

torch::Tensor serial_index(
    torch::Tensor const in,
    torch::Tensor const idx,
    bool const pin_memory = false
);

torch::Tensor to_row_major(torch::Tensor const in);


#endif // FAST_SAMPLER_SAMPLE_AND_SLICE_H_
