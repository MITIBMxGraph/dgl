#include <dgl/base_heterograph.h>
#include <dgl/transform.h>
#include <dgl/array.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/packed_func_ext.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/object.h>
#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/registry.h>
#include <dgl/aten/array_ops.h>
#include <dgl/aten/coo.h>
#include <dgl/aten/csr.h>
#include <dgl/aten/macro.h>
#include <dgl/aten/spmat.h>
#include <dgl/aten/types.h>

#include <vector>
#include <cstdint>

using namespace dgl::runtime;
using namespace dgl::aten;

DGL_REGISTER_GLOBAL("salient.mfg._CAPI_TestMFG")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {

    // std::tie(out_rowptr, out_col, n_ids, out_e_id) = sample_adj(
    //     rowptr, col, std::move(n_ids), n_id_map, size, false, pin_memory);
    // 'SRC/_N' and 'DST/_N'
    const int64_t nvtypes = 2;
    // const int64_t num_dst = subset_size;
    // const int64_t num_src = n_ids.size();

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

    *rv = out_graph_index;
  });


/*

// Returning a vector of MFGs does not work

DGL_REGISTER_GLOBAL("salient.mfg._CAPI_TestMFGVector")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {

    // std::tie(out_rowptr, out_col, n_ids, out_e_id) = sample_adj(
    //     rowptr, col, std::move(n_ids), n_id_map, size, false, pin_memory);
    // 'SRC/_N' and 'DST/_N'
    const int64_t nvtypes = 2;
    // const int64_t num_dst = subset_size;
    // const int64_t num_src = n_ids.size();

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
    std::tuple<dgl::HeteroGraphRef> out_vec = {out_graph_index};

    // *rv = out_graph_index;
    *rv = out_vec;
  });
*/

// HeteroGraphIndexArray
class HeteroGraphArray : public dgl::runtime::Object {
  public:
    explicit HeteroGraphArray(std::vector<dgl::HeteroGraphRef> graphs): graphs(graphs) {}
    virtual ~HeteroGraphArray() = default;

    std::vector<dgl::HeteroGraphRef> graphs;

    /*
    void VisitAttrs(AttrVisitor *v) final {
      v->Visit("graphs", &graphs);
    }
    */

    static constexpr const char* _type_key = "graph.HeteroGraphArray";
    DGL_DECLARE_OBJECT_TYPE_INFO(HeteroGraphArray, dgl::runtime::Object);

  protected:
    HeteroGraphArray(){}
};

DGL_DEFINE_OBJECT_REF(HeteroGraphArrayRef, HeteroGraphArray);
/*
// This is to define a reference class (the wrapper of an object shared pointer).
// A minimal implementation is as follows, but you could define extra methods.
class HeteroGraphArray: public ObjectRef {
 public:
  const HeteroGraphArrayObject* operator->() const {
    return static_cast<const HeteroGraphArrayObject*>(obj_.get());
  }
  using ContainerType = HeteroGraphArrayObject;
};
*/


DGL_REGISTER_GLOBAL("salient.mfg._CAPI_DGLHeteroArrayGetGraphAtIdx")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {
    HeteroGraphArrayRef hga = args[0];
    int idx = args[1];
    *rv = hga->graphs[idx];
});


DGL_REGISTER_GLOBAL("salient.mfg._CAPI_TestDGLHeteroArray")
.set_body([] (dgl::runtime::DGLArgs args, dgl::runtime::DGLRetValue* rv) {

    HeteroGraphArrayRef hgar;

    for (int i = 0; i < 3; i++) {

      const int64_t nvtypes = 2;
      // const int64_t num_dst = subset_size;
      // const int64_t num_src = n_ids.size();

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

      hgar->graphs.push_back(out_graph_index);
    }

    *rv = hgar;

});
