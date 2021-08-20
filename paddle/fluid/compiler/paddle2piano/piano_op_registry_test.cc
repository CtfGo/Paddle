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

#include "paddle/fluid/compiler/paddle2piano/piano_op_registry.h"

#include <unordered_set>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/fluid/compiler/paddle2piano/piano_op_kernel.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace piano {

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
    AddComment(R"DOC(
        Test Operator.

        This operator is used to test piano test op registry OK.

        )DOC");
  }
};

std::unordered_set<note::ElementTypeProto> TestDatatypes() {
  static std::unordered_set<note::ElementTypeProto> supported_types = {
      note::F16, note::F32, note::F64};
  return supported_types;
}

bool TestFilterFunc(Operand* op) {
  // TODO(jiangcheng05) : fill some change of Operand
  return true;
}
}  // namespace piano
}  // namespace paddle

// register paddle op
REGISTER_OP_WITHOUT_GRADIENT(test, paddle::piano::TestOp,
                             paddle::piano::TestOpMaker);
REGISTER_OP_WITHOUT_GRADIENT(op_not_piano, paddle::piano::TestOp,
                             paddle::piano::TestOpMaker);
REGISTER_OP_WITHOUT_GRADIENT(test_limit_backend, paddle::piano::TestOp,
                             paddle::piano::TestOpMaker);

// register backend which support some datetype but all op
REGISTER_PIANO_BACKEND(PIANO_JIT_TEST_TYPE, paddle::piano::TestDatatypes())
// register backend which support some datetype and some op
REGISTER_PIANO_BACKEND(PIANO_JIT_TEST_FILTER, paddle::piano::TestDatatypes(),
                       paddle::piano::TestFilterFunc)

namespace paddle {
namespace piano {

class TestPianoOpMaker : public PianoOpMaker {
 public:
  void Make() override { AddAttr<int>("test_attr", 100); }
};

// register piano op kernel with limit allow backend list
class TestPianoOpWithAllowBackendMaker : public PianoOpMaker {
 public:
  void Make() override {
    SetAllowBackendList({"PIANO_JIT_TEST_FILTER"});
    SetDataTypes(TestDatatypes());
    AddAttr<int>("test_attr", 100);
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

REGISTER_PIANO_OP(test, paddle::piano::TestPianoOpMaker,
                  paddle::piano::TestPianoOpKernel)
REGISTER_PIANO_OP(test_limit_backend,
                  paddle::piano::TestPianoOpWithAllowBackendMaker,
                  paddle::piano::TestPianoOpKernel)

namespace paddle {
namespace piano {

TEST(TestPianoOpRegistry, CheckBackendRegistered) {
  ASSERT_FALSE(PianoOpRegistry::IsBackend("BACKEND_NO_REGISTERED"));
  ASSERT_TRUE(PianoOpRegistry::IsBackend("PIANO_JIT_TEST_TYPE"));
  ASSERT_TRUE(PianoOpRegistry::IsBackend("PIANO_JIT_TEST_FILTER"));

  auto backends = PianoOpRegistry::AllBackendNames();
  std::stable_sort(backends.begin(), backends.end());
  ASSERT_EQ(backends, std::vector<std::string>(
                          {"PIANO_JIT_TEST_FILTER", "PIANO_JIT_TEST_TYPE"}));
  ASSERT_EQ(PianoOpRegistry::BackendDataTypes("PIANO_JIT_TEST_TYPE"),
            TestDatatypes());
}

TEST(TestPianoOpRegistry, CheckPianoOpRegistered) {
  // check piano register OK
  ASSERT_FALSE(PianoOpRegistry::IsPianoOp("op_no_registered"));
  ASSERT_FALSE(PianoOpRegistry::IsPianoOp("op_not_piano"));
  ASSERT_TRUE(PianoOpRegistry::IsPianoOp("test"));
  ASSERT_TRUE(PianoOpRegistry::IsPianoOp("test_limit_backend"));

  // check register store OK
  auto ops = PianoOpRegistry::AllPianoOps();
  std::stable_sort(ops.begin(), ops.end());
  ASSERT_EQ(ops, std::vector<std::string>({"test", "test_limit_backend"}));

  // check piano op's attribute OK
  const auto& attrs = PianoOpRegistry::Attrs("test");
  ASSERT_NE(attrs.find("test_attr"), attrs.cend());

  const auto& attr = attrs.at("test_attr");
  ASSERT_NO_THROW(BOOST_GET_CONST(int, attr));
  ASSERT_EQ(BOOST_GET_CONST(int, attr), 100);

  // check allow backend list OK
  ASSERT_FALSE(PianoOpRegistry::HasAllowBackendList("test"));
  ASSERT_TRUE(PianoOpRegistry::HasAllowBackendList("test_limit_backend"));
  ASSERT_EQ(PianoOpRegistry::AllowBackendList("test_limit_backend"),
            std::vector<std::string>({"PIANO_JIT_TEST_FILTER"}));

  ASSERT_EQ(PianoOpRegistry::PianoOpDataTypes("test_limit_backend"),
            TestDatatypes());
}

TEST(TestPianoOpRegistry, CheckOpKernelRegistered) {
  const auto& kernels = PianoOpRegistry::AllPianoOpKernels("test");

  ASSERT_FALSE(kernels.empty());
}

}  // namespace piano
}  // namespace paddle
