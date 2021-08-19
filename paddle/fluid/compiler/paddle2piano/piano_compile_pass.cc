/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/compiler/paddle2piano/piano_compile_pass.h"
#include "paddle/fluid/framework/ir/subgraph_detector.h"

namespace paddle {

using GraphNodeVec = std::vector<framework::ir::Node*>;
using GraphNodeSet = std::unordered_set<framework::ir::Node*>;

// This interface is used to classify all variables involved in a cluster into
// three types: inputs, outputs, and internals.
static void AnalyseClusterVariables(const GraphNodeVec& cluster,
                                    GraphNodeVec* p_cluster_inputs,
                                    GraphNodeVec* p_cluster_outputs,
                                    GraphNodeVec* p_cluster_internals) {
  GraphNodeSet inputs_recoder;
  GraphNodeSet outputs_recoder;
  for (auto* op_node : cluster) {
    for (auto* input_var_node : op_node->inputs) {
      inputs_recoder.insert(input_var_node);
    }
    for (auto* output_var_node : op_node->outputs) {
      outputs_recoder.insert(output_var_node);
    }
  }
  const GraphNodeSet cluster_set(cluster.begin(), cluster.end());
  for (auto* var_node : outputs_recoder) {
    if (inputs_recoder.count(var_node) > 0) {
      inputs_recoder.erase(var_node);
      bool is_only_used_internal = true;
      for (auto* next_op_node : var_node->outputs) {
        is_only_used_internal &= (cluster_set.count(next_op_node) > 0);
      }
      if (is_only_used_internal) {
        (*p_cluster_internals).push_back(var_node);
      } else {
        (*p_cluster_outputs).push_back(var_node);
      }
    } else {
      (*p_cluster_outputs).push_back(var_node);
    }
  }
  for (auto* var_node : inputs_recoder) {
    (*p_cluster_inputs).push_back(var_node);
  }
}

// Compile a original cluster to generate a compiled cluster, and add links
// between op nodes in the compiled cluster and var nodes in the whole graph.
// The compiled cluster is allowd to reuse some internal var nodes which are
// used in the original cluster.
static GraphNodeVec CompileCluster(const GraphNodeVec& cluster,
                                   const GraphNodeVec& cluster_inputs,
                                   const GraphNodeVec& cluster_outputs,
                                   const GraphNodeVec& cluster_internals,
                                   framework::ir::Graph* graph) {
  GraphNodeVec compiled_cluster;
  // TODO(levi): This is a fake compilation process, we only put one compiled_op
  // in the compiled_cluster. And the real compilation process is under
  // development.
  framework::OpDesc empty_op_desc;
  empty_op_desc.SetType(kPianoCompiledOpName);
  auto* piano_compiled_op_node = graph->CreateOpNode(&empty_op_desc);
  piano_compiled_op_node->inputs = cluster_inputs;
  piano_compiled_op_node->outputs = cluster_outputs;
  for (auto* var_node : cluster_inputs) {
    var_node->outputs.push_back(piano_compiled_op_node);
  }
  for (auto* var_node : cluster_outputs) {
    var_node->inputs.push_back(piano_compiled_op_node);
  }
  compiled_cluster.push_back(piano_compiled_op_node);
  return compiled_cluster;
}

// Check whether a compiled cluster is valid.
static void CheckCompiledClusterValid(
    const GraphNodeVec& compiled_cluster, const GraphNodeVec& cluster_inputs,
    const GraphNodeVec& cluster_outputs,
    const GraphNodeVec& compiled_cluster_inputs,
    const GraphNodeVec& compiled_cluster_outputs) {
  // cluster_inputs and compiled_cluster_inputs must have same contents,
  // and cluster_outputs and compiled_cluster_outputs must have same contents.
  auto same_contents_checker = [](const GraphNodeVec& lhs,
                                  const GraphNodeVec& rhs) -> bool {
    if (lhs.size() != rhs.size()) return false;
    GraphNodeSet lhs_set(lhs.begin(), lhs.end());
    bool is_same = true;
    for (auto* var_node : rhs) {
      is_same &= (lhs_set.count(var_node) > 0);
    }
    return is_same;
  };
  PADDLE_ENFORCE_EQ(
      same_contents_checker(cluster_inputs, compiled_cluster_inputs), true,
      platform::errors::InvalidArgument("The original cluster and the "
                                        "compiled cluster should have same "
                                        "inputs"));
  PADDLE_ENFORCE_EQ(
      same_contents_checker(cluster_outputs, compiled_cluster_outputs), true,
      platform::errors::InvalidArgument("The original cluster and the "
                                        "compiled cluster should have same "
                                        "outputs"));
  // There should be links between var nodes in
  // compiled_cluster_inputs/compiled_cluster_outputs and op nodes in the
  // compiled cluster.
  auto has_link_checker = [](const GraphNodeVec& src,
                             const GraphNodeSet& dest) -> bool {
    bool has_link = false;
    for (auto* op_node : src) {
      has_link |= (dest.count(op_node) > 0);
    }
    return has_link;
  };
  const GraphNodeSet compiled_cluster_set(compiled_cluster.begin(),
                                          compiled_cluster.end());
  for (auto* var_node : compiled_cluster_inputs) {
    PADDLE_ENFORCE_EQ(has_link_checker(var_node->outputs, compiled_cluster_set),
                      true,
                      platform::errors::InvalidArgument(
                          "Each compiled cluster input must have at least one "
                          "link with op node in the compiled cluster"));
  }
  for (auto* var_node : compiled_cluster_outputs) {
    PADDLE_ENFORCE_EQ(has_link_checker(var_node->inputs, compiled_cluster_set),
                      true,
                      platform::errors::InvalidArgument(
                          "Each compiled cluster output must have at least one "
                          "link with op node in the compiled cluster"));
  }
}

// Remove links between op nodes in the original cluster and var nodes in the
// whole graph. Then remove op nodes in the original cluster and unused internal
// var nodes.
static void RemoveUselessLinksAndNodes(
    const GraphNodeVec& cluster, const GraphNodeVec& cluster_internals,
    const GraphNodeVec& compiled_cluster_inputs,
    const GraphNodeVec& compiled_cluster_outputs,
    const GraphNodeVec& compiled_cluster_internals,
    framework::ir::Graph* graph) {
  const GraphNodeSet cluster_set(cluster.begin(), cluster.end());
  auto link_filter = [&cluster_set](GraphNodeVec* p_op_nodes) -> void {
    GraphNodeVec filtered_results;
    for (auto* op_node : *p_op_nodes) {
      if (cluster_set.count(op_node) == 0) {
        filtered_results.emplace_back(op_node);
      }
    }
    (*p_op_nodes).assign(filtered_results.begin(), filtered_results.end());
  };

  for (auto* var_node : compiled_cluster_inputs) {
    link_filter(&(var_node->outputs));
  }
  for (auto* var_node : compiled_cluster_outputs) {
    link_filter(&(var_node->inputs));
  }
  const GraphNodeSet compiled_cluster_internals_set(
      compiled_cluster_internals.begin(), compiled_cluster_internals.end());
  for (auto* var_node : cluster_internals) {
    if (compiled_cluster_internals_set.count(var_node) > 0) {
      link_filter(&(var_node->inputs));
      link_filter(&(var_node->outputs));
    } else {
      graph->RemoveNode(var_node);
    }
  }
  for (auto* op_node : cluster) {
    graph->RemoveNode(op_node);
  }
}

void PianoCompilePass::ApplyImpl(framework::ir::Graph* graph) const {
  // Step1: Detect compile supported subgraphs through SubgraphDetector.
  // Here we call the detected subgraph a cluster.
  PianoOpSearchHelper search_helper;
  auto teller = [&search_helper](const framework::ir::Node* node) -> bool {
    return search_helper.IsSupported(node->Name());
  };
  std::vector<GraphNodeVec> clusters =
      framework::ir::SubgraphDetector(graph, teller)();

  // Step2: Compile and replace each cluster with corresponding compiled
  // cluster.
  for (auto& cluster : clusters) {
    // Analyse variable usage of the original cluster.
    GraphNodeVec cluster_inputs;
    GraphNodeVec cluster_outputs;
    GraphNodeVec cluster_internals;
    AnalyseClusterVariables(cluster, &cluster_inputs, &cluster_outputs,
                            &cluster_internals);

    auto compiled_cluster = CompileCluster(
        cluster, cluster_inputs, cluster_outputs, cluster_internals, graph);

    // Analyse variable usage of the compiled cluster.
    GraphNodeVec compiled_cluster_inputs;
    GraphNodeVec compiled_cluster_outputs;
    GraphNodeVec compiled_cluster_internals;
    AnalyseClusterVariables(compiled_cluster, &compiled_cluster_inputs,
                            &compiled_cluster_outputs,
                            &compiled_cluster_internals);

    CheckCompiledClusterValid(compiled_cluster, cluster_inputs, cluster_outputs,
                              compiled_cluster_inputs,
                              compiled_cluster_outputs);

    RemoveUselessLinksAndNodes(
        cluster, cluster_internals, compiled_cluster_inputs,
        compiled_cluster_outputs, compiled_cluster_internals, graph);
  }
}

}  // namespace paddle

REGISTER_PASS(piano_compile_pass, paddle::PianoCompilePass);
