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

#include <functional>
#include <utility>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/compiler/piano/backends/note_visitor_base.h"
#include "paddle/fluid/compiler/piano/note/function.h"
#include "paddle/fluid/compiler/piano/note/instruction.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/note/opcode.h"
#include "paddle/fluid/compiler/piano/shape.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace piano {
namespace note {

namespace {
class DummyVisitor : public backends::NoteVisitorBase {
 public:
  void VisitConstant(const note::Instruction&) override { state_ = "no"; }
  void VisitParameter(const note::Instruction&) override { state_ = "no"; }
  void VisitBatchNormGrad(const note::Instruction&) override { state_ = "no"; }
  void VisitBatchNormInference(const note::Instruction&) override {
    state_ = "no";
  }
  void VisitBatchNormTraining(const note::Instruction&) override {
    state_ = "no";
  }
  void VisitConvolution(const note::Instruction&) override { state_ = "no"; }
  void VisitDot(const note::Instruction&) override { state_ = "no"; }
  void VisitBroadcast(const note::Instruction&) override { state_ = "no"; }
  void VisitCast(const note::Instruction&) override { state_ = "no"; }
  void VisitCopy(const note::Instruction&) override { state_ = "no"; }
  void VisitExp(const note::Instruction&) override { state_ = "no"; }
  void VisitLog(const note::Instruction&) override { state_ = "no"; }
  void VisitNegative(const note::Instruction&) override { state_ = "no"; }
  void VisitNot(const note::Instruction&) override { state_ = "no"; }
  void VisitReshape(const note::Instruction&) override { state_ = "no"; }
  void VisitReverse(const note::Instruction&) override { state_ = "no"; }
  void VisitRsqrt(const note::Instruction&) override { state_ = "no"; }
  void VisitSlice(const note::Instruction&) override { state_ = "no"; }
  void VisitSqrt(const note::Instruction&) override { state_ = "no"; }
  void VisitTranspose(const note::Instruction&) override { state_ = "no"; }
  void VisitAdd(const note::Instruction&) override { state_ = "yes"; }
  void VisitAnd(const note::Instruction&) override { state_ = "no"; }
  void VisitCompare(const note::Instruction&) override { state_ = "no"; }
  void VisitDivide(const note::Instruction&) override { state_ = "no"; }
  void VisitMaximum(const note::Instruction&) override { state_ = "no"; }
  void VisitMinimum(const note::Instruction&) override { state_ = "no"; }
  void VisitMultiply(const note::Instruction&) override { state_ = "no"; }
  void VisitOr(const note::Instruction&) override { state_ = "no"; }
  void VisitSubtract(const note::Instruction&) override { state_ = "no"; }
  void VisitXor(const note::Instruction&) override { state_ = "no"; }
  void VisitSelect(const note::Instruction&) override { state_ = "no"; }
  void VisitConcatenate(const note::Instruction&) override { state_ = "no"; }
  void VisitReduce(const note::Instruction&) override { state_ = "no"; }
  void VisitRng(const note::Instruction&) override { state_ = "no"; }
  void VisitSort(const note::Instruction&) override { state_ = "no"; }
  void VisitTuple(const note::Instruction&) override { state_ = "no"; }

  const std::string& state() { return state_; }

 private:
  std::string state_{"unknown"};
};

struct GetAttrFunctor {
  GetAttrFunctor(const MapType& attrs, const std::string& name)
      : attrs_(attrs), name_(name) {}

  template <typename T>
  void apply() {
    boost::get<T>(attrs_.at(name_));
  }

 private:
  const MapType& attrs_;
  std::string name_;
};
}  // namespace

class IrTest : public testing::Test {
 protected:
  virtual void SetUp() {
    // input shapes
    const Shape arg1_shape(note::F32, {3, 6});
    const Shape arg2_shape(note::F32, {3, 6});
    // output shape
    const Shape result_shape(note::F32, {3, 6});
    // function signature
    const Signature signature({arg1_shape, arg2_shape}, {"arg1.1", "arg2.2"},
                              result_shape);
    SignatureProto signature_proto = signature.ToProto();
    signature_proto_.Swap(&signature_proto);

    // set instr1_proto_
    instr1_proto_.set_name("arg1.1");
    instr1_proto_.set_opcode(GetOpName(OpCode::kParameter));
    instr1_proto_.set_id(1);
    instr1_proto_.set_parameter_number(0);
    *instr1_proto_.mutable_shape() = arg1_shape.ToProto();
    auto* attrs1_map = instr1_proto_.mutable_attrs();
    AttrValueProto val1_proto;
    val1_proto.set_d(3.141);
    attrs1_map->insert(ProtoMapType::value_type("test_double", val1_proto));
    auto* strings = val1_proto.mutable_strings()->mutable_value();
    *strings->Add() = "hello";
    *strings->Add() = "world";
    attrs1_map->insert(ProtoMapType::value_type("test_strings", val1_proto));
    auto* bools = val1_proto.mutable_bools()->mutable_value();
    bools->Add(true);
    bools->Add(false);
    attrs1_map->insert(ProtoMapType::value_type("test_bools", val1_proto));
    auto* ints = val1_proto.mutable_ints()->mutable_value();
    ints->Add(8);
    ints->Add(26);
    attrs1_map->insert(ProtoMapType::value_type("test_ints", val1_proto));

    // set instr2_proto_
    instr2_proto_.set_name("arg2.2");
    instr2_proto_.set_opcode(GetOpName(OpCode::kParameter));
    instr2_proto_.set_id(2);
    instr2_proto_.set_parameter_number(0);
    *instr2_proto_.mutable_shape() = arg2_shape.ToProto();
    auto* attrs2_map = instr2_proto_.mutable_attrs();
    AttrValueProto val2_proto;
    val2_proto.set_b(true);
    attrs2_map->insert(ProtoMapType::value_type("test_bool", val2_proto));
    auto* longs = val2_proto.mutable_longs()->mutable_value();
    longs->Add(8l);
    longs->Add(16l);
    attrs2_map->insert(ProtoMapType::value_type("test_longs", val2_proto));
    auto* floats = val2_proto.mutable_floats()->mutable_value();
    floats->Add(8.6f);
    floats->Add(7.6f);
    attrs2_map->insert(ProtoMapType::value_type("test_floats", val2_proto));
    auto* doubles = val2_proto.mutable_doubles()->mutable_value();
    doubles->Add(5.66);
    doubles->Add(6.66);
    attrs2_map->insert(ProtoMapType::value_type("test_doubles", val2_proto));

    // set instr3_proto_
    instr3_proto_.set_name("add.3");
    instr3_proto_.set_opcode(GetOpName(OpCode::kAdd));
    instr3_proto_.set_id(3);
    instr3_proto_.set_parameter_number(2);
    *instr3_proto_.mutable_shape() = result_shape.ToProto();
    instr3_proto_.add_operand_ids(1);
    instr3_proto_.add_operand_ids(2);
    auto* attrs3_map = instr3_proto_.mutable_attrs();
    AttrValueProto val3_proto;
    val3_proto.set_s("Add");
    attrs3_map->insert(ProtoMapType::value_type("test_string", val3_proto));
    val3_proto.set_i(-1);
    attrs3_map->insert(ProtoMapType::value_type("test_int", val3_proto));
    val3_proto.set_l(-100l);
    attrs3_map->insert(ProtoMapType::value_type("test_long", val3_proto));
    val3_proto.set_f(-1.414f);
    attrs3_map->insert(ProtoMapType::value_type("test_float", val3_proto));

    // set func_proto_
    func_proto_.set_name(func_name_);
    *func_proto_.mutable_signature() = signature_proto_;
    func_proto_.set_return_id(instr3_proto_.id());
    function_id_ = instr3_proto_.id() + 1;
    func_proto_.set_id(function_id_);
    *func_proto_.add_instructions() = instr1_proto_;
    *func_proto_.add_instructions() = instr2_proto_;
    *func_proto_.add_instructions() = instr3_proto_;
  }

 protected:
  std::string func_name_{"union_12510013719728903619"};
  FunctionProto func_proto_;
  std::int64_t function_id_;
  SignatureProto signature_proto_;
  InstructionProto instr1_proto_;
  InstructionProto instr2_proto_;
  InstructionProto instr3_proto_;
};

TEST_F(IrTest, FunctionToString) {
  std::unordered_map<std::int64_t, Function*> func_index;
  Function func(func_proto_, func_index);
  std::string func_str = func.ToString();
  std::string expected_str =
      R"RES(def %union_12510013719728903619(arg1.1: f32[3, 6]{}, arg2.2: f32[3, 6]{}) -> f32[3, 6]{} {
  %arg1.1 = f32[3, 6]{} parameter(), test_bools={true, false}, test_double=3.141000_d, test_ints={8_i, 26_i}, test_strings={"hello", "world"}
  %arg2.2 = f32[3, 6]{} parameter(), test_bool=true, test_doubles={5.660000_d, 6.660000_d}, test_floats={8.600000_f, 7.600000_f}, test_longs={8_l, 16_l}
  return %add.3 = f32[3, 6]{} add(f32[3, 6]{} %arg1.1, f32[3, 6]{} %arg2.2), test_float=-1.414000_f, test_int=-1_i, test_long=-100_l, test_string="Add"
})RES";
  ASSERT_EQ(expected_str, func_str);
  LOG(INFO) << "Function string:\n" << func_str;
}

TEST_F(IrTest, FunctionToProto) {
  std::unordered_map<std::int64_t, Function*> func_index;
  Function func(func_proto_, func_index);
  std::string expected_str = func_proto_.DebugString();
  std::string real_str = func.ToProto().DebugString();
  ASSERT_EQ(expected_str, real_str);
  LOG(INFO) << "The function prototext:\n" << real_str;
}

TEST_F(IrTest, FunctionDetails) {
  std::unordered_map<std::int64_t, Function*> func_index;
  Function func(func_proto_, func_index);
  ASSERT_EQ(func.name(), func_name_);
  ASSERT_EQ(func.signature().ToProto().DebugString(),
            signature_proto_.DebugString());
  ASSERT_EQ(func.global_id(), function_id_);
  ASSERT_EQ(func.return_instr()->opcode(), OpCode::kAdd);
  ASSERT_EQ(func.params_num(), 2);
  ASSERT_EQ(func.param_instrs().size(), 2);
  ASSERT_EQ(func.param_instr(0)->opcode(), OpCode::kParameter);
  ASSERT_EQ(func.param_instr(1)->opcode(), OpCode::kParameter);
  ASSERT_EQ(func.instruction(2)->opcode(), OpCode::kAdd);
  ASSERT_EQ(func.instructions()[2]->opcode(), OpCode::kAdd);
}

TEST_F(IrTest, InstructionDetails) {
  std::unordered_map<std::int64_t, Function*> func_index;
  Function func(func_proto_, func_index);
  const Instruction* instr = func.return_instr();
  ASSERT_EQ(instr->name(), "add.3");
  ASSERT_EQ(instr->parent()->name(), func_name_);
  ASSERT_EQ(instr->operands().size(), 2);
  ASSERT_EQ(
      instr->operand(0)->GetAttr<std::vector<std::string>>("test_strings")[0],
      "hello");
  ASSERT_EQ(instr->operand(0)->global_id(), 1);
  ASSERT_EQ(instr->operand(0)->parameter_number(), 0);
  ASSERT_EQ(instr->operand(1)->ctrl_predecessors().size(), 0);
  ASSERT_EQ(instr->operand(1)->ctrl_successors().size(), 0);
  ASSERT_EQ(instr->operand(1)->parameter_number(), 0);
  ASSERT_EQ(boost::get<std::int64_t>(instr->attrs().at("test_long")), -100l);
  ASSERT_EQ(instr->shape().Rank(), 2);
  DummyVisitor visitor;
  instr->Accept(&visitor);
  ASSERT_EQ(visitor.state(), "yes");
  Instruction* mutable_instr = func.mutable_instruction(0);
  mutable_instr->Accept(&visitor);
  ASSERT_EQ(visitor.state(), "no");
}

TEST_F(IrTest, VisitAttr) {
  std::unordered_map<std::int64_t, Function*> func_index;
  Function func(func_proto_, func_index);
  const auto& cpp_map = func.return_instr()->attrs();
  const auto& proto_map = instr3_proto_.attrs();

  for (auto it = proto_map.begin(); it != proto_map.end(); it++) {
    GetAttrFunctor getter(cpp_map, it->first);
    EXPECT_NO_THROW(VisitAttrType(it->second.value_case(), getter));
  }
}

}  // namespace note
}  // namespace piano
}  // namespace paddle
