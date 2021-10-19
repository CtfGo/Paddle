// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <map>
#include <memory>
#include <string>
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/runtime/cinn_runtime.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

static constexpr char kX[] = "X";
static constexpr char kOutputs[] = "Out";
static constexpr char kSubGraphHashKey[] = "subgraph_hashkey";

using CinnScope = cinn::hlir::framework::Scope;
using CinnRuntimeProgram = cinn::hlir::framework::Program;

namespace details {
// Initialize all variables used on CINN compiled program, and return
// a map with relation: variable name (Paddle side) --> variable object
static std::map<std::string, framework::Variable*> InitializeVariables(
    const framework::Scope& paddle_scope, const CinnScope& cinn_compiled_scope,
    const std::map<std::string, std::string>& variable_name_pd2cinn,
    framework::Scope* temp_scope) {
  std::map<std::string, framework::Variable*> variable_name2ptr;

  // determine which variables are already defined in Paddle.
  std::unordered_set<std::string> initialized_cinn_names;
  for (const auto& name_pd2cinn : variable_name_pd2cinn) {
    const auto& pd_name = name_pd2cinn.first;
    const auto& cinn_name = name_pd2cinn.second;
    // Some variables defined in Paddle maybe eliminated by CINN optimized
    // passes.
    if (!cinn_compiled_scope->FindVar(cinn_name)) {
      VLOG(2) << "Variable(" << pd_name << ") has been eliminated by CINN";
      continue;
    }
    auto* var_ptr = Scope.FindVar(pd_name);
    PADDLE_ENFORCE_NOT_NULL(
        var_ptr, platform::errors::NotFound("Variable(%s) not found in Scope.",
                                            pd_name));
    PADDLE_ENFORCE_EQ(
        var_ptr->IsType<framework::LoDTensor>(), true,
        platform::errors::InvalidArgument(
            "Variable(%s) is not LoDTensor that is only supported now.",
            pd_name));

    const auto& cinn_tensor = cinn_compiled_scope->GetTensor(cinn_name);
    auto dim_from_cinn = framework::make_ddim(cinn_tensor.shape().data());
    if (var_ptr->Get<LoDTensor>().numel() != 0) {
      PADDLE_ENFORCE_EQ(
          var_ptr->Get<framework::LoDTensor>().dims(), dim_from_cinn,
          platform::errors::PreconditionNotMet(
              "The Tensor(%s)'s shape infered by Paddle is not equal to CINN, "
              "Paddle shape is [%s] but CINN is [%s].",
              pd_name, var_ptr->Get<framework::LoDTensor>().dims(),
              dim_from_cinn));
    } else {
      VLOG(2) << "The shape of variable(" << pd_name << ") is infered by CINN";
      var_ptr->GetMutable<framework::LoDTensor>()->Resize(dim_from_cinn);
    }

    variable_name2ptr.emplace(pd_name, var_ptr);
    initialized_cinn_names.insert(cinn_name);
  }

  // Initialize variables that are needed in CINN in addition,
  // temporary variables in other words.
  for (const auto& name_view : cinn_compiled_scope->var_names()) {
    std::string cinn_name(name_view.data(), name_view.size());
    if (initialized_cinn_names.count(cinn_name)) {
      // variable is defined in paddle and has been collected in above.
      continue;
    }
    VLOG(2) << "Create a temporary variable(%s) for launching CINN execution";

    const auto& cinn_tensor = cinn_compiled_scope->GetTensor(cinn_name);
    auto dim_from_cinn = framework::DDim(cinn_tensor.shape().data().data(),
                                         cinn_tensor.shape().size());
    auto* var_ptr = temp_scope->Var(cinn_name);
    auto* paddle_tensor = var_ptr->GetMutable<LoDTensor>();
    paddle_tensor->Resize(dim_from_cinn);
    // use the same name with cinn
    variable_name2ptr.emplace(cinn_name, var_ptr);
  }

  return variable_name2ptr;
}

// CINN variable name --> cinn_pod_value_t
static std::map<std::string, cinn_pod_value_t> PackExecutionArgs(
    const std::map<std::string, framework::Variable*>& all_variables,
    const std::map<std::string, std::string>& variable_name_pd2cinn,
    std::vector<std::unique_ptr<cinn_buffer_t>>* hold_buffers) {
  std::map<std::string, cinn_pod_value_t> execution_arguments;
  for (const auto& name2ptr : all_variables) {
    const auto& pd_name = name2ptr.first;
    auto* var_ptr = name2ptr.second;
    const auto& cinn_name = variable_name_pd2cinn.count(pd_name)
                                ? variable_name_pd2cinn.at(pd_name)
                                : pd_name;

    const auto& tensor = var_ptr->Get<framework::LoDTensor>();
    std::unique_ptr<cinn_buffer_t> buffer_ptr(new cinn_buffer_t());
    buffer_ptr->resize(
        reinterpret_cast<const cinn_dimension_t*>(shape.data().data()),
        shape.size());
    buffer_ptr->memory = reinterpret_cast<uint8_t*>(tensor.data<float>());
    execution_arguments.emplace(cinn_name, buffer_ptr->get());
    hold_buffers->emplace_back(std::move(buffer_ptr));
  }

  return execution_arguments;
}

}  // namespace details

class CinnLaunchOp : public framework::OperatorBase {
 public:
  CinnLaunchOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    VLOG(2) << "CinnLaunchOp RunImpl";
    // Step 1. Get subgraph object and prepare input
    PADDLE_ENFORCE_EQ(HasAttr(kSubGraphHashKey), true,
                      platform::errors::NotFound(
                          "No Attribute(%s) found for CinnLaunchOp operator.",
                          kSubGraphHashKey));
    const auto& subgraph_key = Attr<std::string>(kSubGraphHashKey);
    // TODO(CtfGo): updated after related interface ready, using local object
    // temporarily
    ir::Graph temp_graph;
    auto* subgraph = &temp_graph;
    // auto* subgraph = CinnRunner::GetGraph(subgraph_key);
    PADDLE_ENFORCE_NOT_NULL(
        subgraph, platform::errors::NotFound(
                      "Subgraph with hashkey(%s) not found in CinnRunner.",
                      subgraph_key));

    OP_INOUT_CHECK(HasInputs(kX), "Input", kX, "CinnLaunchOp");
    std::map<std::string, const framework::LoDTensor*> feed_targets;
    for (const auto& var_name : Inputs(kX)) {
      auto* var_ptr = scope.FindVar(var_name);
      PADDLE_ENFORCE_EQ(
          var_ptr->IsType<framework::LoDTensor>(), true,
          platform::errors::InvalidArgument(
              "Variable(%s) is not LoDTensor that is only supported now.",
              var_name));
      feed_targets.emplace(var_name, &(var_ptr->Get<LoDTensor>()));
    }

    // Step 2. Compile subgraph into an executable CinnRuntimeProgram object
    // TODO(CtfGo): using local object temporarily,
    // will be replaced after related interface ready
    std::shared_ptr<CinnScope> cinn_compiled_scope;
    // std::vector<std::unique_ptr<Instruction>> instructions;
    std::unique_ptr<CinnRuntimeProgram> cinn_runtime_program(
        new CinnRuntimeProgram(cinn_compiled_scope, {}));
    std::map<std::string, std::string> variable_name_pd2cinn;

    // Step 3. Initialize all variables used in the compiled program
    auto temp_scope = scope.NewTmpScope();
    auto variable_name2ptr = details::InitializeVariables(
        scope, *cinn_compiled_scope, variable_name_pd2cinn, temp_scope->get());

    for (const auto& name2ptr : variable_name2ptr) {
      const auto& var_name = name2ptr.first;
      auto* var_ptr = name2ptr.second;
      if (!var_ptr->Get<framework::LoDTensor>().IsInitialized()) {
        // TODO(CtfGo): support mutable corresponding c++ type with the CINN
        // Type
        auto* tensor =
            var_ptr->GetMutable<framework::LoDTensor>()->mutable_data<float>(
                place);
        VLOG(2) << "Variable(" << var_name
                << ") allocates buffer on CinnLaunchOp.";
      }
    }

    // check all outpout variables is included in arguments delivered
    for (const auto& var_name : Outputs(kOutputs)) {
      PADDLE_ENFORCE_GT(
          variable_name2ptr.count(var_name),
          0 platform::errors::NotFound("The output variable(%s) must be "
                                       "included in variables passed to CINN",
                                       var_name));
    }

    // Step 4. Pack execution arguments and launch CINN
    //        to execute the computation of subgraph.
    std::vector<std::unique_ptr<cinn_buffer_t>> hold_buffers;
    auto arguments_name2object = details::PackExecutionArgs(
        variable_name2ptr, variable_name_pd2cinn, &hold_buffers);

    cinn_runtime_program->Execute(&arguments_name2objct);
  }
};

class CinnLaunchOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(kX,
             "(vector<LoDTensor>)"
             "which are the input of subgraph inside the CinnLaunchOp.")
        .AsDuplicable();
    AddOutput(kOutputs,
              "(vector<LoDTensor>)"
              "which will be assigned with values generated by the "
              "subgraph and used by succeeding operators of the CinnLaunchOp");
    .AsDuplicable();
    AddAttr<std::string>(
        kSubGraphHashKey,
        "(string)"
        "the hash key of serialized string of the subgraph, which "
        "will be deserialized to ir::Graph and use it to launch cinn");
    AddComment(R"DOC(
CinnLaunch Operator.

This operator is used to launch CINN(https://github.com/PaddlePaddle/CINN/blob/develop/README.md)
to compile a graph and execute the compiled object.

Input of this operator is a set of variables which are used by the subgraph
and output is a also a set of variables which are updated or generated by the graph,
and kSubGraphHashKey should be set necessarily to get the Graph object.

It accomplishs the computation of subgraph following several steps:
1. Fetch ir::Graph object from CinnRunner using kkSubGraphHashKey.
2. Compile the graph to a compiled object, and insert it to the
global cache so that we can directly query it from this cache next time
when shape of input variables are not all changed.
3. Create and instantiate all variables needed by the subgraph using
the info(type,shape) included in the compiled object.
4. Pack each tensor buffer of all variables needed by cinn as execution arguments.
5. Launch execution of cinn runtime program with above arguments,
the result would be output by writing value on underlying buffer address.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    cinn_launch, ops::CinnLaunchOp, ops::CinnLaunchOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>);
