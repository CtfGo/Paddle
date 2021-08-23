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

#include "paddle/fluid/compiler/paddle2piano/piano_op_kernel_context.h"

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/fluid/compiler/paddle2piano/piano_op_kernel.h"
#include "paddle/fluid/compiler/paddle2piano/piano_op_registry.h"
#include "paddle/fluid/compiler/paddle2piano/piano_scope.h"
#include "paddle/fluid/compiler/piano/note_builder.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace piano {

std::unordered_set<note::ElementTypeProto> TestDatatypes() {
  static std::unordered_set<note::ElementTypeProto> supported_types = {
      note::F16, note::F32, note::F64};
  return supported_types;
}

using paddle::framework::InferShapeContext;
using paddle::framework::OpProtoAndCheckerMaker;
using paddle::framework::OperatorWithKernel;

class TestOp : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

  void InferShape(InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "test");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "test");

    auto in_dims = ctx->GetInputDim("X");

    ctx->SetOutputDim("Out", in_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class TestOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of test op.");
    AddOutput("Out", "(Tensor), The output tensor of test op.");
    AddAttr<float>("scale", "scale attribute of test op");
    AddAttr<int>("same", "same name attribute of test op");
    AddComment(R"DOC(
        Test Operator.

        This operator is used to test piano test op registry OK.

        )DOC");
  }
};

// register piano op kernel with limit allow backend list
class TestPianoOpMaker : public PianoOpMaker {
 public:
  void Make() override {
    SetAllowBackendList({"PIANO_JIT_TEST_TYPE"});
    SetDataTypes(TestDatatypes());
    AddAttr<int>("same", 1);
  }
};

class TestPianoOpKernel : public PianoOpKernel {
 public:
  void Compile(const PianoOpKernelContext& context) const override {
    // do nothing, pass
  }
};

}  // namespace piano
}  // namespace paddle

// register backend which support some datetype but all op
REGISTER_PIANO_BACKEND(PIANO_JIT_TEST_TYPE, paddle::piano::TestDatatypes())

// register paddle op
REGISTER_OP_WITHOUT_GRADIENT(test, paddle::piano::TestOp,
                             paddle::piano::TestOpMaker);

REGISTER_PIANO_OP(test, paddle::piano::TestPianoOpMaker,
                  paddle::piano::TestPianoOpKernel)

namespace paddle {
namespace piano {

TEST(PianoContextTest, scope) {
  PianoScope scope;

  // nothing added
  std::string empty_op = "empty";
  ASSERT_FALSE(scope.HasLocalOperand(empty_op));
  ASSERT_FALSE(scope.HasOperand(empty_op));
  ASSERT_EQ(scope.parent(), nullptr);
  ASSERT_ANY_THROW(scope.GetLocalOperand(empty_op));
  ASSERT_ANY_THROW(scope.GetOperand(empty_op));
  ASSERT_EQ(scope.FindScope(empty_op), nullptr);

  // add operand
  std::string name1 = "op1";
  Operand op1;
  scope.SetOperand(name1, op1);

  ASSERT_TRUE(scope.HasLocalOperand(name1));
  ASSERT_TRUE(scope.HasOperand(name1));
  ASSERT_NO_THROW(scope.GetLocalOperand(name1));
  ASSERT_NO_THROW(scope.GetOperand(name1));
  ASSERT_EQ(scope.FindScope(name1), &scope);
  ASSERT_EQ(scope.LocalOperandNames(), std::vector<std::string>({name1}));

  // temp scope
  auto tmp_scope = scope.NewTmpScope();
  ASSERT_EQ(tmp_scope->parent(), &scope);
  ASSERT_FALSE(scope.HasKid(tmp_scope.get()));

  std::string name2 = "op2";
  Operand op2;
  tmp_scope->SetOperand(name2, op2);

  ASSERT_FALSE(scope.HasLocalOperand(name2));
  ASSERT_FALSE(scope.HasOperand(name2));

  // add sub-scope
  auto* sub_scope = scope.NewScope();
  ASSERT_EQ(sub_scope->parent(), &scope);
  ASSERT_TRUE(scope.HasKid(sub_scope));

  ASSERT_FALSE(sub_scope->HasLocalOperand(name1));
  ASSERT_TRUE(sub_scope->HasOperand(name1));
  ASSERT_NO_THROW(sub_scope->GetOperand(name1));
  ASSERT_EQ(sub_scope->FindScope(name1), &scope);

  // erase sub-scope
  ASSERT_TRUE(scope.EraseKid(sub_scope));
  ASSERT_FALSE(scope.HasKid(sub_scope));
}

TEST(PianoContextTest, basic) {
  // create OpDesc
  framework::ProgramDesc program;
  auto* global_block = program.MutableBlock(0);

  auto* op = global_block->AppendOp();
  op->SetType("test");
  op->SetInput("X", {"IN1"});
  op->SetOutput("Out", {"OUT1"});
  op->SetAttr("scale", 3.14f);
  op->SetAttr("same", 0);

  // create scope and NoteBuilder
  PianoScope scope;
  Operand op_x;
  scope.SetOperand("X", op_x);

  NoteBuilder builder("test_expand");

  // create PianoOpKernelContext
  PianoOpKernelContext context(op, &scope, &builder);
  const PianoOpKernelContext& ctx = context;

  // basic test
  ASSERT_EQ(ctx.Type(), "test");
  ASSERT_EQ(ctx.Builder(), &builder);

  // test input
  ASSERT_TRUE(ctx.HasInput("X"));
  // Operand no match for 'operator=='
  ASSERT_NO_THROW(ctx.GetInput("X"));
  ASSERT_FALSE(ctx.HasInput("Y"));
  ASSERT_ANY_THROW(ctx.GetInput("Y"));

  // test output
  Operand op_out;
  ASSERT_NO_THROW(ctx.SetOutput("Out", op_out));
  ASSERT_ANY_THROW(ctx.SetOutput("Y", op_out));

  // test attribute
  ASSERT_EQ(ctx.DataTypes(), TestDatatypes());

  ASSERT_TRUE(ctx.HasAttr("scale"));
  ASSERT_EQ(ctx.GetAttr<float>("scale"), 3.14f);

  ASSERT_TRUE(ctx.HasAttr("same"));
  ASSERT_EQ(ctx.GetAttr<int>("same"), 1);
}

}  // namespace piano
}  // namespace paddle
