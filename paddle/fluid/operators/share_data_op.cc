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

#include "paddle/fluid/operators/share_data_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class ShareDataOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "ShareData");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ShareData");
    auto in_type = ctx->GetInputsVarType("Input")[0];
    auto out_type = ctx->GetOutputsVarType("Out")[0];

    PADDLE_ENFORCE_EQ(
        in_type == framework::proto::VarType::LOD_TENSOR ||
            in_type == framework::proto::VarType::SELECTED_ROWS,
        true,
        platform::errors::InvalidArgument(
            "Type of Variable[Input] must be LoDTensor or SelectedRows!"));
    PADDLE_ENFORCE_EQ(
        in_type, out_type,
        platform::errors::InvalidArgument(
            "The type of input (Input) and output (Out) are inconsistent."));

    ctx->ShareDim("Input", "Out");
  }
};

class ShareDataOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "The input tensor.");
    AddOutput("Out",
              "The returned tensor, will share data with the input Tensor.");
    AddComment(R"DOC(
ShareData Operator.

Return a tensor that share data with the input tensor and
always doesn't have a Tensor copy.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    share_data, ops::ShareDataOp, ops::ShareDataOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(share_data, ops::ShareDataKernel<bool>,
                       ops::ShareDataKernel<int>, ops::ShareDataKernel<int8_t>,
                       ops::ShareDataKernel<uint8_t>,
                       ops::ShareDataKernel<paddle::platform::float16>,
                       ops::ShareDataKernel<int64_t>,
                       ops::ShareDataKernel<float>,
                       ops::ShareDataKernel<double>)
