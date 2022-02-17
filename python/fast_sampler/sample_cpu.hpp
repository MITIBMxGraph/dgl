#pragma once

#include <random>
#include <tuple>

#include <torch/torch.h>

#include "parallel_hashmap/phmap.h"
#include "utils.hpp"

// includes for nvtx profiling
#include "nvToolsExt.h"

// likley too many includes here, were needed for development
#include <dgl/base_heterograph.h>
#include <dgl/transform.h>
#include <dgl/array.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/packed_func_ext.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/object.h>
#include <dgl/runtime/c_runtime_api.h>
#include <dgl/aten/array_ops.h>

// just inluced the rest ha
#include <dgl/aten/coo.h>
#include <dgl/aten/csr.h>
#include <dgl/aten/macro.h>
#include <dgl/aten/spmat.h>
#include <dgl/aten/types.h>

#include <vector>
#include <tuple>
#include <utility>
//#include "../../array/cpu/array_utils.h"
// TODO: CLEANUP
#include "/home/gridsan/pmurzynowski/dgl/src/array/cpu/array_utils.h"

thread_local std::mt19937 gen;

inline auto get_initial_sample_adj_hash_map(const std::vector<int64_t>& n_ids) {
  phmap::flat_hash_map<int64_t, int64_t> n_id_map;
  for (size_t i = 0; i < n_ids.size(); ++i) {
    n_id_map[n_ids[i]] = i;
  }
  return n_id_map;
}

// using SingleSample = std::
//     tuple<torch::Tensor, torch::Tensor, std::vector<int64_t>, torch::Tensor>;
using SingleSample = std::
    tuple<dgl::IdArray, dgl::IdArray, std::vector<int64_t>, dgl::IdArray>;

// Returns `rowptr`, `col`, `n_id`, `e_id`
inline SingleSample sample_adj(
    torch::Tensor rowptr,
    torch::Tensor col,
    std::vector<int64_t> n_ids,
    phmap::flat_hash_map<int64_t, int64_t>& n_id_map,
    int64_t num_neighbors,
    bool replace,
    bool pin_memory = false) {
  const auto idx_size = n_ids.size();

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  const auto col_data = col.data_ptr<int64_t>();

  // intermediate for transpose
  auto tmp_rowptr = torch::empty(idx_size + 1, rowptr.options().pinned_memory(false));
  const auto tmp_rowptr_data = tmp_rowptr.data_ptr<int64_t>();
  tmp_rowptr_data[0] = 0;
  // adjacency vector of (col, e_id)
  std::vector<std::vector<std::tuple<int64_t, int64_t>>> cols(idx_size);

  const auto expand_neighborhood = [&](auto add_neighbors) -> void {
    for (size_t i = 0; i < idx_size; ++i) {
      const auto n = n_ids[i];
      const auto row_start = rowptr_data[n];
      const auto row_end = rowptr_data[n + 1];
      const auto neighbor_count = row_end - row_start;
      
      const auto add_neighbor = [&](const int64_t p) -> void {
        const auto e = row_start + p;
        const auto c = col_data[e];

        auto ins = n_id_map.insert({c, n_ids.size()});
        if (ins.second) {
          n_ids.push_back(c);
        }
        cols[i].push_back(std::make_tuple(ins.first->second, e));
      };

      add_neighbors(neighbor_count, add_neighbor);
      tmp_rowptr_data[i + 1] = tmp_rowptr_data[i] + cols[i].size();
    }
  };

  if (num_neighbors < 0) { // No sampling ======================================
    expand_neighborhood([](const int64_t neighbor_count, auto add_neighbor) {
      for (int64_t j = 0; j < neighbor_count; j++) {
        add_neighbor(j);
      }
    });
  } else if (replace) { // Sample with replacement =============================
    expand_neighborhood(
        [num_neighbors](const int64_t neighbor_count, auto add_neighbor) {
          if (neighbor_count <= 0) return;
          for (int64_t j = 0; j < num_neighbors; j++) {
            add_neighbor(gen() % neighbor_count);
          }
        });
  } else { // Sample without replacement via Robert Floyd algorithm ============
    std::vector<int64_t> perm;
    perm.reserve(num_neighbors);
    expand_neighborhood([num_neighbors, &perm](
                            const int64_t neighbor_count, auto add_neighbor) {
      perm.clear();


      if (neighbor_count <= num_neighbors) {
        for (int64_t j = 0; j < neighbor_count; j++) {
          add_neighbor(j);
        }
      } else { // See: https://www.nowherenearithaca.com/2013/05/
               //      robert-floyds-tiny-and-beautiful.html
        for (int64_t j = neighbor_count - num_neighbors; j < neighbor_count; j++) {
          const int64_t option = gen() % j;
          auto winner = option;
          if (std::find(perm.cbegin(), perm.cend(), option) == perm.cend()) {
            perm.push_back(option);
            winner = option;
          } else {
            perm.push_back(j);
            winner = j;
          }

          add_neighbor(winner);
        }
      }
    });
  }


  const auto E = tmp_rowptr_data[idx_size];

  // intermediate variables for transpose
  auto tmp_col = torch::empty(E, col.options().pinned_memory(false));
  const auto tmp_col_data = tmp_col.data_ptr<int64_t>();
  auto tmp_e_id = torch::empty(E, col.options().pinned_memory(false));
  const auto tmp_e_id_data = tmp_e_id.data_ptr<int64_t>();

  {
    size_t i = 0;
    for (auto& col_vec : cols) {
      // do not need to sort when performing transpose operation
      // std::sort(
      //     col_vec.begin(),
      //     col_vec.end(),
      //     [](const auto& a, const auto& b) -> bool {
      //       return std::get<0>(a) < std::get<0>(b);
      //     });
      for (const auto& value : col_vec) {
        tmp_col_data[i] = std::get<0>(value);
        tmp_e_id_data[i] = std::get<1>(value);
        i += 1;
      }
    }
  }

  // output
  const auto n_col = n_id_map.size();
  // printf("(should be on stack) n_col: %p\n", &n_col);
  /*
  auto out_rowptr = torch::empty(n_col + 1, rowptr.options().pinned_memory(pin_memory));
  const auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();
  // attempt with contiguous memory
  // const auto out_rowptr_data = out_rowptr.contiguous().data_ptr<int64_t>();
  auto out_col = torch::empty(E, col.options().pinned_memory(pin_memory));
  const auto out_col_data = out_col.data_ptr<int64_t>();
  auto out_e_id = torch::empty(E, col.options().pinned_memory(pin_memory));
  const auto out_e_id_data = out_e_id.data_ptr<int64_t>();
  */
  // using dgl
  constexpr DLContext ctx = DLContext{kDLCPU, 0};
  // 64 bit
  const uint8_t nbits = 64;
  dgl::IdArray out_rowptr = dgl::aten::NewIdArray(n_col+1, ctx, nbits);
  const auto out_rowptr_data = out_rowptr.Ptr<int64_t>();
  dgl::IdArray out_col = dgl::aten::NewIdArray(E, ctx, nbits);
  const auto out_col_data = out_col.Ptr<int64_t>();
  dgl::IdArray out_e_id = dgl::aten::NewIdArray(E, ctx, nbits);
  const auto out_e_id_data = out_e_id.Ptr<int64_t>();


  // transpose (from scipy)
  // https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h#L378-L423
  const auto nnz = E;
  const auto n_row = idx_size;
  std::fill(out_rowptr_data, out_rowptr_data + n_col, 0);
  // don't need to sort above because of this but perhaps has better cache behavior if sort
  for (auto n = 0; n < nnz; n++) { 
    out_rowptr_data[tmp_col_data[n]]++;
  }
  for (auto col = 0, cumsum = 0; col < n_col; col++) { 
    auto temp  = out_rowptr_data[col];
    out_rowptr_data[col] = cumsum;
    cumsum += temp;
  }
  out_rowptr_data[n_col] = nnz; 
  for (auto row = 0; row < n_row; row++) {
    for (auto jj = tmp_rowptr_data[row]; jj < tmp_rowptr_data[row+1]; jj++) {
      auto col  = tmp_col_data[jj];
      auto dest = out_rowptr_data[col];
      out_col_data[dest] = row;
      out_e_id_data[dest] = tmp_e_id_data[jj];
      out_rowptr_data[col]++;
    }
  }
  for (auto col = 0, last = 0; col <= n_col; col++) {
    auto temp = out_rowptr_data[col];
    out_rowptr_data[col] = last;
    last = temp;
  }
  // DEBUG
  // printf("out_rowptr: %p\n", out_rowptr);
  // printf("&out_rowptr: %p\n", &out_rowptr);
  // printf("out_rowptr_data: %p\n", out_rowptr_data);

  return std::make_tuple(
      std::move(out_rowptr),
      std::move(out_col),
      std::move(n_ids),
      std::move(out_e_id));
}

inline SingleSample sample_adj(
    torch::Tensor rowptr,
    torch::Tensor col,
    std::vector<int64_t> n_ids,
    int64_t num_neighbors,
    bool replace,
    bool pin_memory = false) {
  auto n_id_map = get_initial_sample_adj_hash_map(n_ids);
  return sample_adj(
      std::move(rowptr),
      std::move(col),
      std::move(n_ids),
      n_id_map,
      num_neighbors,
      replace,
      pin_memory);
}

//inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
// not tested after switching to dgl
inline std::tuple<dgl::IdArray, dgl::IdArray, torch::Tensor, dgl::IdArray>
sample_adj(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor idx,
    int64_t num_neighbors,
    bool replace,
    bool pin_memory = false) {
  const auto idx_data = idx.data_ptr<int64_t>();
  auto res = sample_adj(
      std::move(rowptr),
      std::move(col),
      {idx_data, idx_data + idx.numel()},
      num_neighbors,
      replace,
      pin_memory);
  auto& n_ids = std::get<2>(res);
  return std::make_tuple(
      std::move(std::get<0>(res)),
      std::move(std::get<1>(res)),
      vector_to_tensor(n_ids),
      std::move(std::get<3>(res)));
}
