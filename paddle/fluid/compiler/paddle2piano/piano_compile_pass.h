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

#pragma once

#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {

constexpr char kPianoCompiledOpName[] = "piano_compiled_op_name";

/*
 * A pass named piano_compile_pass, the function of this pass is:
 *
 * a) Detect the subgraphs that can be compiled by the piano compiler. We call a
 * detected subgraph a cluster, which is consisted of several op nodes.
 *
 * b) Call the piano compiler to compile each original cluster and get the
 * compiled cluster, which is consisted of several PianoCompiledOps.
 *
 * c) Replace the original cluster with corresponding compiled cluster on the
 * original graph.
 *
 * In this pass, some questions are handled with cautions:
 *
 * a) How to determine whether two op nodes can be divided into a cluster?
 * Firstly, both op nodes should be compile supported.
 * Secondly, there should be a direct path between the two op nodes through a
 * var node.
 * Thirdly, there should be no extral path between the two op nodes through
 * unsupported op nodes.
 * Lastly, if op nodes a and b can be divied into a cluster, op nodes b and c
 * can be devided into a cluster, a and c can also be devided into a cluster.
 * The implementation of cluster detection is enclosured in class
 * SubGraphDetector.
 *
 * b) How to deal with the links between the var nodes in global graph and the
 * op nodes in a cluster?
 * We first add links between the var nodes in global graph and the op nodes in
 * the compiled cluster, and then remove useless links between the var nodes in
 * global graph and the op nodes in the original cluster.
 *
 * c) Whether to allow the compiled cluster to reuse internal var nodes in the
 * original cluster?
 * During the replacement of the cluster, a loose strategy is adopted on whether
 * the internal var nodes of the original cluster can be reused. The compiled
 * cluster is allowed to reuse some internal var nodes in the original cluster,
 * and it is also allowed to generate a new internal var node.
 */
class PianoCompilePass : public framework::ir::Pass {
 protected:
  void ApplyImpl(framework::ir::Graph* graph) const override;
};

// TODO(levi): Remove this helper when the piano compilation support information
// has been registered in the operator definition.
class PianoOpSearchHelper {
 private:
  // TODO(levi): need to fill supported ops list.
  std::unordered_set<std::string> supported_op_names_ = {
      "relu", "elementwise_add", "mul",
  };

 public:
  bool IsSupported(const std::string& op_name) {
    return supported_op_names_.count(op_name) > 0;
  }
};

}  // namespace paddle
