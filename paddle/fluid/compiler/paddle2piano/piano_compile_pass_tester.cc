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
#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {

static void CheckInputsAndOutputs(
    const framework::ir::Node* piano_compiled_op,
    const std::unordered_set<std::string>& golden_input_var_names,
    const std::unordered_set<std::string>& golden_output_var_names) {
  for (const auto var_node : piano_compiled_op->inputs) {
    EXPECT_EQ(golden_input_var_names.count(var_node->Name()), 1UL);
  }
  for (const auto var_node : piano_compiled_op->outputs) {
    EXPECT_EQ(golden_output_var_names.count(var_node->Name()), 1UL);
  }
}

std::unique_ptr<framework::ir::Graph> BuildGraphBasic(bool backward = false) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (x, y)                     mul              -> tmp_0
  // (tmp_0, z)                 elementwise_add  -> tmp_1
  // tmp_1                      relu             -> tmp_2
  // (tmp_2, w)                 elementwise_add  -> tmp_3
  //
  // Expression: tmp_3 = relu(mul(x, y) + z) + w
  framework::ir::Layers layers;
  std::vector<int64_t> shape = {16, 32};
  auto* x = layers.data("x", {16, 16});
  auto* y = layers.data("y", {16, 32});
  auto* tmp_0 = layers.mul(x, y);
  auto* z = layers.data("z", shape);
  auto* tmp_1 = layers.elementwise_add(tmp_0, z);
  auto* tmp_2 = layers.relu(tmp_1);
  auto* w = layers.data("w", shape);
  auto* tmp_3 = layers.elementwise_add(tmp_2, w);
  std::vector<framework::VarDesc*> elementwise_vars = {tmp_0, tmp_1, tmp_2,
                                                       tmp_3};
  for (auto* var : elementwise_vars) {
    var->SetShape(shape);
  }

  if (backward) {
    layers.backward({tmp_3});
  }

  std::unique_ptr<framework::ir::Graph> graph(
      new framework::ir::Graph(layers.main_program()));
  for (auto* n : graph->Nodes()) {
    if (n && n->IsVar() && n->Var()) {
      n->Var()->SetDataType(framework::proto::VarType::FP32);
    }
  }
  return std::move(graph);
}

std::unique_ptr<framework::ir::Graph> BuildGraphMultiClusters(
    bool backward = false) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (x, y)                     mul              -> tmp_0
  // (tmp_0, z)                 elementwise_add  -> tmp_1
  // (m, n)                     mul              -> tmp_2
  // (tmp_1, tmp_2)             elementwise_fake -> tmp_3
  // tmp_3                      relu             -> tmp_4
  // (tmp_4, w)                 elementwise_fake -> tmp_5
  // (tmp_4, tmp_5)             elementwise_add  -> tmp_6
  // (tmp_6, y)                 elementwise_add  -> tmp_7
  // (tmp_7, n)                 elementwise_add  -> tmp_8
  //
  // Expression: tmp_8 = relu(fake(mul(x, y) + z, mul(m, n))) +
  //                     fake(relu(fake(mul(x, y) + z, mul(m, n))), w) + y + n
  framework::ir::Layers layers;
  std::vector<int64_t> shape = {16, 32};
  auto* x = layers.data("x", {16, 16});
  auto* y = layers.data("y", {16, 32});
  auto* tmp_0 = layers.mul(x, y);
  auto* z = layers.data("z", shape);
  auto* tmp_1 = layers.elementwise_add(tmp_0, z);
  auto* m = layers.data("m", {16, 16});
  auto* n = layers.data("n", {16, 32});
  auto* tmp_2 = layers.mul(m, n);
  auto* tmp_3 = layers.elementwise_fake(tmp_1, tmp_2);
  auto* tmp_4 = layers.relu(tmp_3);
  auto* w = layers.data("w", shape);
  auto* tmp_5 = layers.elementwise_fake(tmp_4, w);
  auto* tmp_6 = layers.elementwise_add(tmp_4, tmp_5);
  auto* tmp_7 = layers.elementwise_add(tmp_6, y);
  auto* tmp_8 = layers.elementwise_add(tmp_7, n);
  std::vector<framework::VarDesc*> elementwise_vars = {
      tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7, tmp_8};
  for (auto* var : elementwise_vars) {
    var->SetShape(shape);
  }

  if (backward) {
    layers.backward({tmp_3});
  }

  std::unique_ptr<framework::ir::Graph> graph(
      new framework::ir::Graph(layers.main_program()));
  for (auto* n : graph->Nodes()) {
    if (n && n->IsVar() && n->Var()) {
      n->Var()->SetDataType(framework::proto::VarType::FP32);
    }
  }
  return std::move(graph);
}

TEST(PianoCompilePass, basic) {
  std::unique_ptr<framework::ir::Graph> graph = BuildGraphBasic(false);
  auto pass = framework::ir::PassRegistry::Instance().Get("piano_compile_pass");
  VLOG(3) << framework::ir::DebugString(graph);
  graph.reset(pass->Apply(graph.release()));
  VLOG(3) << framework::ir::DebugString(graph);

  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "relu"), 0);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "elementwise_add"), 0);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "mul"), 0);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, kPianoCompiledOpName), 1);

  const framework::ir::Node* piano_compiled_op =
      framework::ir::GetOpNodes(graph, kPianoCompiledOpName).at(0);
  EXPECT_EQ(piano_compiled_op->inputs.size(), 4UL);
  EXPECT_EQ(piano_compiled_op->outputs.size(), 1UL);
  CheckInputsAndOutputs(piano_compiled_op, {"x", "y", "z", "w"}, {"tmp_3"});
}

TEST(PianoCompilePass, basic_with_backward) {
  std::unique_ptr<framework::ir::Graph> graph = BuildGraphBasic(true);
  auto pass = framework::ir::PassRegistry::Instance().Get("piano_compile_pass");
  VLOG(3) << framework::ir::DebugString(graph);
  graph.reset(pass->Apply(graph.release()));
  VLOG(3) << framework::ir::DebugString(graph);

  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "relu"), 0);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "elementwise_add"), 0);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "mul"), 0);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "relu_grad"), 1);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "elementwise_add_grad"), 2);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "mul_grad"), 1);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, kPianoCompiledOpName), 1);

  const framework::ir::Node* piano_compiled_op =
      framework::ir::GetOpNodes(graph, kPianoCompiledOpName).at(0);
  EXPECT_EQ(piano_compiled_op->inputs.size(), 4UL);
  EXPECT_EQ(piano_compiled_op->outputs.size(), 4UL);
  CheckInputsAndOutputs(piano_compiled_op, {"x", "y", "z", "w"},
                        {"tmp_0", "tmp_1", "tmp_2", "tmp_3"});
}

TEST(PianoCompilePass, multi_clusters) {
  std::unique_ptr<framework::ir::Graph> graph = BuildGraphMultiClusters(false);
  auto pass = framework::ir::PassRegistry::Instance().Get("piano_compile_pass");
  VLOG(3) << framework::ir::DebugString(graph);
  graph.reset(pass->Apply(graph.release()));
  VLOG(3) << framework::ir::DebugString(graph);

  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "relu"), 0);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "elementwise_add"), 0);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "mul"), 0);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "elementwise_fake"), 2);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, kPianoCompiledOpName), 4);

  // NOTE(levi): This test case is carefully designed, the number of inputs of
  // the piano_compiled_ops should be exactly 1, 2, 3 and 4.
  const std::vector<framework::ir::Node*> piano_compiled_ops =
      framework::ir::GetOpNodes(graph, kPianoCompiledOpName);
  std::unordered_set<int> inputs_number_recoder{1, 2, 3, 4};
  for (const auto piano_compiled_op : piano_compiled_ops) {
    int inputs_number = piano_compiled_op->inputs.size();
    EXPECT_EQ(inputs_number_recoder.count(inputs_number), 1UL);
    inputs_number_recoder.erase(inputs_number);
    switch (inputs_number) {
      case 1:
        EXPECT_EQ(piano_compiled_op->outputs.size(), 1UL);
        CheckInputsAndOutputs(piano_compiled_op, {"tmp_3"}, {"tmp_4"});
        break;
      case 2:
        EXPECT_EQ(piano_compiled_op->outputs.size(), 1UL);
        CheckInputsAndOutputs(piano_compiled_op, {"m", "n"}, {"tmp_2"});
        break;
      case 3:
        EXPECT_EQ(piano_compiled_op->outputs.size(), 1UL);
        CheckInputsAndOutputs(piano_compiled_op, {"x", "y", "z"}, {"tmp_1"});
        break;
      case 4:
        EXPECT_EQ(piano_compiled_op->outputs.size(), 1UL);
        CheckInputsAndOutputs(piano_compiled_op, {"tmp_4", "tmp_5", "y", "n"},
                              {"tmp_8"});
        break;
      default:
        break;
    }
  }
}

TEST(PianoCompilePass, multi_clusters_with_backward) {
  std::unique_ptr<framework::ir::Graph> graph = BuildGraphMultiClusters(true);
  auto pass = framework::ir::PassRegistry::Instance().Get("piano_compile_pass");
  VLOG(3) << framework::ir::DebugString(graph);
  graph.reset(pass->Apply(graph.release()));
  VLOG(3) << framework::ir::DebugString(graph);

  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "relu"), 0);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "elementwise_add"), 0);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "mul"), 0);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "elementwise_fake"), 2);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "relu_grad"), 1);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "elementwise_add_grad"), 4);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "mul_grad"), 2);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, "elementwise_fake_grad"), 2);
  EXPECT_EQ(framework::ir::GetNumOpNodes(graph, kPianoCompiledOpName), 4);

  // NOTE(levi): This test case is carefully designed, the number of inputs of
  // the piano_compiled_ops should be exactly 1, 2, 3 and 4.
  const std::vector<framework::ir::Node*> piano_compiled_ops =
      framework::ir::GetOpNodes(graph, kPianoCompiledOpName);
  std::unordered_set<int> inputs_number_recoder{1, 2, 3, 4};
  for (const auto piano_compiled_op : piano_compiled_ops) {
    int inputs_number = piano_compiled_op->inputs.size();
    EXPECT_EQ(inputs_number_recoder.count(inputs_number), 1UL);
    inputs_number_recoder.erase(inputs_number);
    switch (inputs_number) {
      case 1:
        EXPECT_EQ(piano_compiled_op->outputs.size(), 1UL);
        CheckInputsAndOutputs(piano_compiled_op, {"tmp_3"}, {"tmp_4"});
        break;
      case 2:
        EXPECT_EQ(piano_compiled_op->outputs.size(), 1UL);
        CheckInputsAndOutputs(piano_compiled_op, {"m", "n"}, {"tmp_2"});
        break;
      case 3:
        EXPECT_EQ(piano_compiled_op->outputs.size(), 2UL);
        CheckInputsAndOutputs(piano_compiled_op, {"x", "y", "z"},
                              {"tmp_0", "tmp_1"});
        break;
      case 4:
        EXPECT_EQ(piano_compiled_op->outputs.size(), 3UL);
        CheckInputsAndOutputs(piano_compiled_op, {"tmp_4", "tmp_5", "y", "n"},
                              {"tmp_6", "tmp_7", "tmp_8"});
        break;
      default:
        break;
    }
  }
}

}  // namespace paddle

USE_PASS(piano_compile_pass);
