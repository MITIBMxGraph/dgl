#include "fs_sample_and_slice.hpp"

ProtoSample multilayer_sample(
    std::vector<int64_t> n_ids,
    std::vector<int64_t> const& sizes,
    torch::Tensor rowptr,
    torch::Tensor col,
    bool pin_memory = false) {
  auto n_id_map = get_initial_sample_adj_hash_map(n_ids);
  Blocks blocks;
  blocks.reserve(sizes.size());
  for (auto size : sizes) {
    auto const subset_size = n_ids.size();
    //torch::Tensor out_rowptr, out_col, out_e_id;
    dgl::IdArray out_rowptr, out_col, out_e_id;

    // sample_adj outputs a relation graph
    // for all intents and purposes this is the MFG for that layer
    std::tie(out_rowptr, out_col, n_ids, out_e_id) = sample_adj(
        rowptr, col, std::move(n_ids), n_id_map, size, false, pin_memory);
    // 'SRC/_N' and 'DST/_N'
    const int64_t nvtypes = 2;
    const int64_t num_dst = subset_size;
    const int64_t num_src = n_ids.size();
    std::vector<dgl::SparseFormat> formats_vec = {dgl::ParseSparseFormat("csr")};
    const auto code = SparseFormatsToCode(formats_vec);
    auto hgptr = dgl::CreateFromCSR(nvtypes, num_src, num_dst,
                                    std::move(out_rowptr), std::move(out_col), std::move(out_e_id),
                                    code); 
    auto rel_graph = dgl::HeteroGraphRef(hgptr);

    // create metagraph
    constexpr DLContext ctx = DLContext{kDLCPU, 0};
    const uint8_t nbits = 64;
    // currently suming one type of relation graph, so the meta graph has only two nodes
    const int64_t num_nodes = 2;
    // src_ids contains node 0 of metagraph
    const int64_t num_src_ids = 1;
    dgl::IdArray src_ids = dgl::aten::NewIdArray(num_src_ids, ctx, nbits);
    src_ids.Ptr<int64_t>()[0] = 0;
    // dst_ids contains node 1 of metagraph
    const int64_t num_dst_ids = 1;
    dgl::IdArray dst_ids = dgl::aten::NewIdArray(num_dst_ids, ctx, nbits);
    dst_ids.Ptr<int64_t>()[0] = 1;
    // making readonly, so immutable
    auto metagraph = dgl::GraphRef(dgl::ImmutableGraph::CreateFromCOO(num_nodes, src_ids, dst_ids));

    // combine relation graph with metagraph
    // only have on relation graph and most simple metagraph
    std::vector<int64_t> num_nodes_per_type = {num_src, num_dst};
    std::vector<dgl::HeteroGraphPtr> rel_ptrs = {rel_graph.sptr()};
    auto out_hgptr = CreateHeteroGraph(metagraph.sptr(), rel_ptrs, num_nodes_per_type);
    auto out_graph_index = dgl::HeteroGraphRef(out_hgptr);

    blocks.emplace_back(out_graph_index);
  }

  std::reverse(blocks.begin(), blocks.end());
  return {vector_to_tensor(n_ids), std::move(blocks)};
}


ProtoSample multilayer_sample(
    torch::Tensor idx,
    std::vector<int64_t> const& sizes,
    torch::Tensor rowptr,
    torch::Tensor col,
    bool pin_memory = false) {
  const auto idx_data = idx.data_ptr<int64_t>();
  return multilayer_sample(
      {idx_data, idx_data + idx.numel()},
      sizes,
      std::move(rowptr),
      std::move(col),
      pin_memory);
}


template <typename scalar_t>
torch::Tensor serial_index_impl(
    torch::Tensor const in,
    torch::Tensor const idx,
    int64_t const n,
    bool const pin_memory = false) {
  const auto f = in.sizes().back();
  TORCH_CHECK(
      (in.strides().size() == 2 && in.strides().back() == 1) ||
          (in.sizes().back() == 1),
      "input must be 2D row-major tensor");

  torch::Tensor out =
      torch::empty({n, f}, in.options().pinned_memory(pin_memory));
  auto inptr = in.data_ptr<scalar_t>();
  auto outptr = out.data_ptr<scalar_t>();
  auto idxptr = idx.data_ptr<int64_t>();
  for (int64_t i = 0; i < std::min(idx.numel(), n); ++i) {
    const auto row = idxptr[i];
    std::copy_n(inptr + row * f, f, outptr + i * f);
  }

  return out;
}

template <typename scalar_t>
torch::Tensor serial_index_impl(
    torch::Tensor const in,
    torch::Tensor const idx,
    bool const pin_memory = false) {
  return serial_index_impl<scalar_t>(in, idx, idx.numel(), pin_memory);
}

torch::Tensor serial_index(
    torch::Tensor const in,
    torch::Tensor const idx,
    int64_t const n,
    bool const pin_memory = false) {
  return AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, in.scalar_type(), "serial_index", [&] {
    return serial_index_impl<scalar_t>(in, idx, n, pin_memory);
  });
}

torch::Tensor serial_index(
    torch::Tensor const in,
    torch::Tensor const idx,
    bool const pin_memory = false) {
  return AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, in.scalar_type(), "serial_index", [&] {
    return serial_index_impl<scalar_t>(in, idx, pin_memory);
  });
}

torch::Tensor to_row_major(torch::Tensor const in) {
  TORCH_CHECK(in.strides().size() == 2, "only support 2D tensors");
  auto const tr = in.sizes().front();
  auto const tc = in.sizes().back();

  if (in.strides().front() == tc && in.strides().back() == 1) {
    return in; // already in row major
  }

  TORCH_CHECK(
      in.strides().front() == 1 && tr == in.strides().back(),
      "input has unrecognizable stides");

  auto out = torch::empty_strided(in.sizes(), {tc, 1}, in.options());

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::Long, in.scalar_type(), "to_row_major", [&] {
        auto inptr = in.data_ptr<scalar_t>();
        auto outptr = out.data_ptr<scalar_t>();

        for (int64_t r = 0; r < tr; ++r) {
          for (int64_t c = 0; c < tc; ++c) {
            outptr[r * tc + c] = inptr[c * tr + r];
          }
        }
      });

  return out;
}




void fast_sampler_thread(FastSamplerSlot& slot) {

  // annotatex with nvtx
  nvtxRangePushA("fast_sampler_thread");

  std::unique_lock<decltype(slot.mutex)> lock(slot.mutex);
  while (true) {
    if (slot.should_hibernate()) {
      slot.cv.wait(lock, [&slot] {
        return slot.should_decommission() || !slot.should_hibernate();
      });
    }

    if (slot.should_decommission()) {
      return;
    }

    std::pair<int32_t, int32_t> pair;
    if (!slot.session->inputs.try_dequeue_from_producer(
            slot.session->iptok, pair)) {
      // std::this_thread::yield();
      continue;
    }

    slot.session->items_in_queue.acquire();

    // check if we were woken just to decommission or hibernate
    if (slot.should_hibernate() || slot.should_decommission()) {
      continue;
    }

    auto const this_batch_size = pair.second - pair.first;

    auto const& config = slot.session->config;
    const auto idx_data = config.idx.data_ptr<int64_t>();
    nvtxRangePushA("multilayer_sample");
    auto proto = multilayer_sample(
        {idx_data + pair.first, idx_data + pair.second},
        config.sizes,
        config.rowptr,
        config.col,
        config.pin_memory);
    nvtxRangePop();
    auto const& n_id = proto.first;
    nvtxRangePushA("slicing");
    // printf("Slicing x\n");
    // std::cout << config.x.sizes() << std::endl;
    // std::cout << n_id.sizes() << std::endl;
    // std::cout << torch::max(n_id) << std::endl;
    // std::cout << torch::min(n_id) << std::endl;
    auto x_s = serial_index(config.x, n_id, config.pin_memory);
    std::optional<torch::Tensor> y_s;
    if (config.y.has_value()) {
      // printf("Slicing y\n");
      y_s = serial_index(*config.y, n_id, this_batch_size, config.pin_memory);
    }
    nvtxRangePop();
    /*
    nvtxRangePushA("toDGLBlock");
    toDGLBlock(proto);
    nvtxRangePop();
    */

    // TODO: Implement limit on the size of the output queue,
    //       to avoid high memory consumption when outpacing the training code.
    slot.session->outputs.enqueue(
        slot.optok, {std::move(x_s), std::move(y_s), std::move(proto.second), std::move(pair)});
    ++slot.session->num_inserted_batches;
  }

  nvtxRangePop();
}

FastSamplerThread thread_factory() {
  return FastSamplerThread{fast_sampler_thread};
}

/*
// repliate using dgl's ffi
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<FastSamplerConfig>(m, "Config")
      .def(py::init<>())
      .def_readwrite("x", &FastSamplerConfig::x)
      .def_readwrite("y", &FastSamplerConfig::y)
      .def_readwrite("rowptr", &FastSamplerConfig::rowptr)
      .def_readwrite("col", &FastSamplerConfig::col)
      .def_readwrite("idx", &FastSamplerConfig::idx)
      .def_readwrite("batch_size", &FastSamplerConfig::batch_size)
      .def_readwrite("sizes", &FastSamplerConfig::sizes)
      .def_readwrite(
          "skip_nonfull_batch", &FastSamplerConfig::skip_nonfull_batch)
      .def_readwrite("pin_memory", &FastSamplerConfig::pin_memory);
  py::class_<FastSamplerSession>(m, "Session")
      .def(
          py::init<size_t, unsigned int, FastSamplerConfig>(),
          py::arg("num_threads"),
          py::arg("max_items_in_queue"),
          py::arg("config"))
      .def_readonly("config", &FastSamplerSession::config)
      .def("try_get_batch", &FastSamplerSession::try_get_batch)
      .def("blocking_get_batch", &FastSamplerSession::blocking_get_batch)
      .def_property_readonly(
          "num_consumed_batches", &FastSamplerSession::get_num_consumed_batches)
      .def_property_readonly(
          "num_total_batches", &FastSamplerSession::get_num_total_batches)
      .def_property_readonly(
          "approx_num_complete_batches",
          &FastSamplerSession::get_approx_num_complete_batches)
      .def_readonly("total_blocked_dur", &FastSamplerSession::total_blocked_dur)
      .def_readonly(
          "total_blocked_occasions",
          &FastSamplerSession::total_blocked_occasions);
}
*/
