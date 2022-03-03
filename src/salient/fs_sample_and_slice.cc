#include "fs_sample_and_slice.hpp"

ProtoSample multilayer_sample(
    std::vector<int64_t> n_ids,
    std::vector<int64_t> const& sizes,
    torch::Tensor rowptr,
    torch::Tensor col,
    bool pin_memory) {
  auto n_id_map = get_initial_sample_adj_hash_map(n_ids);


  auto blocks = std::make_shared<HeteroGraphArray>();
  blocks->graphs.reserve(sizes.size());

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

    blocks->graphs.emplace_back(out_graph_index);
  }

  std::reverse(blocks->graphs.begin(), blocks->graphs.end());
  ProtoSample rv;
  rv.indices = vector_to_tensor(n_ids);
  rv.mfgs = HeteroGraphArrayRef(blocks);
  return rv;
}


ProtoSample multilayer_sample(
    torch::Tensor idx,
    std::vector<int64_t> const& sizes,
    torch::Tensor rowptr,
    torch::Tensor col,
    bool pin_memory) {
  const auto idx_data = idx.data_ptr<int64_t>();
  return multilayer_sample(
      {idx_data, idx_data + idx.numel()},
      sizes,
      std::move(rowptr),
      std::move(col),
      pin_memory);
}


template <typename scalar_t>
dgl::NDArray serial_index_impl(
    torch::Tensor const in,
    torch::Tensor const idx,
    int64_t const n,
    bool const pin_memory) {
  const auto f = in.sizes().back();
  TORCH_CHECK(
      (in.strides().size() == 2 && in.strides().back() == 1) ||
          (in.sizes().back() == 1),
      "input must be 2D row-major tensor");

  auto inptr = in.data_ptr<scalar_t>();
  auto idxptr = idx.data_ptr<int64_t>();

  constexpr DLContext ctx = DLContext{kDLCPU, 0};
  const uint8_t nbits = sizeof(scalar_t) * 8;
  dgl::NDArray out = dgl::NDArray::Empty({n, f}, DLDataType{kDLInt, nbits, 1}, ctx);
  const auto outptr = out.Ptr<scalar_t>();


  for (int64_t i = 0; i < std::min(idx.numel(), n); ++i) {
    const auto row = idxptr[i];
    std::copy_n(inptr + row * f, f, outptr + i * f);
  }

  return out;
}

template <typename scalar_t>
dgl::NDArray serial_index_impl(
    torch::Tensor const in,
    torch::Tensor const idx,
    bool const pin_memory) {
  return serial_index_impl<scalar_t>(in, idx, idx.numel(), pin_memory);
}

dgl::NDArray serial_index(
    torch::Tensor const in,
    torch::Tensor const idx,
    int64_t const n,
    bool const pin_memory) {
  return AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, in.scalar_type(), "serial_index", [&] {
    return serial_index_impl<scalar_t>(in, idx, n, pin_memory);
  });
}

dgl::NDArray serial_index(
    torch::Tensor const in,
    torch::Tensor const idx,
    bool const pin_memory) {
  return AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, in.scalar_type(), "serial_index", [&] {
    return serial_index_impl<scalar_t>(in, idx, pin_memory);
  });
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
    auto const& n_id = proto.indices;
    nvtxRangePushA("slicing");
    // printf("Slicing x\n");
    // std::cout << config.x.sizes() << std::endl;
    // std::cout << n_id.sizes() << std::endl;
    // std::cout << torch::max(n_id) << std::endl;
    // std::cout << torch::min(n_id) << std::endl;
    auto x_s = serial_index(config.x, n_id, config.pin_memory);
    optional<dgl::NDArray> y_s;
    // std::optional
    // if (config.y.has_value()) {
    // boost::optional
    if (config.y) {
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
        slot.optok, {std::move(x_s), std::move(y_s), std::move(proto.mfgs), std::move(pair)});
    ++slot.session->num_inserted_batches;
  }

  nvtxRangePop();
}

FastSamplerThread thread_factory() {
  return FastSamplerThread{fast_sampler_thread};
}
