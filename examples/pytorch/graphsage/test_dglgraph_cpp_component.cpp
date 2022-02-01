#include <pybind11/pybind11.h>

// dgl imports
#include <dgl/graph.h>
#include <dgl/base_heterograph.h>

int add(int i, int j) {
    return i + j;
}

//int accept_graph(dgl::Graph graph) {
int accept_graph(dgl::HeteroGraphRef graph) {
  printf("accepted graph\n");
  return 0;
}

PYBIND11_MODULE(test_dglgraph_cpp_component, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
    m.def("accept_graph", &accept_graph, "Test function to print DGLGraph");
}
