#include "fs_common.hpp"
#include "fs_return_types.hpp"


/*
 * Not valid
DGL_REGISTER_GLOBAL("salient._CAPI_TestNone")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    int x = args[0];
    if (x >= 0) {
      *rv = x * 100;
    } else {
      *rv = boost::none;
    }
});
*/


dgl::HeteroGraphRef createTestHeteroGraphRef() {
  const int64_t nvtypes = 2;
  const int64_t num_dst = 2;
  const int64_t num_src = 8;
  std::vector<int64_t> optr = {0, 0, 0, 1, 2, 3, 4, 5, 6};
  std::vector<int64_t> ocol = {0, 0, 0, 1, 1, 1};
  std::vector<int64_t> oeid = {15439, 15460, 15461, 15593, 15597, 15590};
  dgl::IdArray out_rowptr = VecToIdArray(optr, 64);
  dgl::IdArray out_col = VecToIdArray(ocol, 64);
  dgl::IdArray out_e_id = VecToIdArray(oeid, 64);

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
  return out_graph_index;
}


HeteroGraphArrayRef createTestHeteroGraphArrayRef() {
    auto hga = std::make_shared<HeteroGraphArray>();
    for (int i = 0; i < 3; i++) {
      hga->graphs.push_back(createTestHeteroGraphRef());
    }
    return HeteroGraphArrayRef(hga);
}


OptionalPreparedSampleRef createTestOptionalPreparedSample() {
    auto ops = std::make_shared<OptionalPreparedSample>();
    PreparedSample ps;
    ps.mfgs = createTestHeteroGraphArrayRef();
    ps.x = torch::ones({2, 2});
    ps.y = torch::ones({2, 1});
    ps.range = std::make_pair(30, 40);
    ops->value = ps;
    return OptionalPreparedSampleRef(ops);
}


DGL_REGISTER_GLOBAL("salient._CAPI_TestMFG")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    *rv = createTestHeteroGraphRef();
  });


DGL_REGISTER_GLOBAL("salient._CAPI_TestDGLHeteroArray")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    *rv = createTestHeteroGraphArrayRef();
});

DGL_REGISTER_GLOBAL("salient._CAPI_TestOptionalPreparedSample")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    *rv = createTestOptionalPreparedSample();
});
