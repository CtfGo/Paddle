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

#include "paddle/fluid/compiler/piano/symbolization/meta_op.h"
#include <utility>
#include "gtest/gtest.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/note/opcode.h"

namespace paddle {
namespace piano {
namespace symbolization {

TEST(MetaOpTest, TestParameter) {
  NoteBuilder builder("test_parameter");
  ASSERT_THROW(Parameter(&builder, -1, Shape(note::F32, {}), "arg"),
               paddle::platform::EnforceNotMet);
  auto param_op = Parameter(&builder, 1, Shape(note::F32, {1, 2}), "arg");
  ASSERT_EQ(&builder, param_op.Builder());
  EXPECT_TRUE(param_op.Valid());
  EXPECT_EQ(Shape(note::F32, {1, 2}), param_op.Shape());
}

TEST(MetaOpTest, TestConstantD0) {
  NoteBuilder builder("test_constant_d0");
  auto constant_d0_op = ConstantD0<int32_t>(&builder, 110);
  ASSERT_EQ(&builder, constant_d0_op.Builder());
  EXPECT_TRUE(constant_d0_op.Valid());
  EXPECT_EQ(Shape(note::S32, {}), constant_d0_op.Shape());

  auto&& module_proto = builder.Build();
  ASSERT_EQ(1, module_proto.functions_size());
  const auto& entry_proto = module_proto.functions(0);
  ASSERT_EQ(1, entry_proto.instructions_size());
  EXPECT_EQ(note::GetOpName(note::OpCode::kConstant),
            entry_proto.instructions(0).opcode());
  const auto& constant_instr = entry_proto.instructions(0);
  ASSERT_EQ(1, constant_instr.attrs().size());
  const auto& attr_value = constant_instr.attrs().at(note::kConstantValue);
  ASSERT_TRUE(attr_value.has_i());
  EXPECT_EQ(110, attr_value.i());
}

TEST(MetaOpTest, TestBroadcast) {
  NoteBuilder builder("test_broadcast");
  // add dimensions_alignment automatically when it is empty
  auto param_op1 = Parameter(&builder, 0, Shape(note::F32, {1, 2}), "arg1");
  auto broadcast_op1 = Broadcast(param_op1, {3, 5, 2});
  ASSERT_EQ(&builder, broadcast_op1.Builder());
  ASSERT_TRUE(broadcast_op1.Valid());
  EXPECT_EQ(Shape(note::F32, {3, 5, 2}), broadcast_op1.Shape());
  // complete parameters
  auto param_op2 = Parameter(&builder, 1, Shape(note::F32, {3, 2}), "arg2");
  auto broadcast_op2 = Broadcast(param_op2, {3, 5, 2}, {0, 2});
  ASSERT_EQ(&builder, broadcast_op2.Builder());
  ASSERT_TRUE(broadcast_op2.Valid());
  EXPECT_EQ(Shape(note::F32, {3, 5, 2}), broadcast_op2.Shape());
  // check build result
  auto&& module_proto = builder.Build();
  ASSERT_EQ(1, module_proto.functions_size());
  const auto& entry_proto = module_proto.functions(0);
  ASSERT_EQ(4, entry_proto.instructions_size());
  EXPECT_EQ(note::GetOpName(note::OpCode::kBroadcast),
            entry_proto.instructions(1).opcode());
  EXPECT_EQ(note::GetOpName(note::OpCode::kBroadcast),
            entry_proto.instructions(3).opcode());
  // check dimensions_alignment of broadcast instruction
  const auto& broadcast_instr1 = entry_proto.instructions(1);
  ASSERT_EQ(1, broadcast_instr1.attrs().size());
  const auto& attr_value1 =
      broadcast_instr1.attrs().at(note::kBroadcastAlignment);
  ASSERT_TRUE(attr_value1.has_longs());
  EXPECT_EQ(2, attr_value1.longs().value_size());
  EXPECT_EQ(1, attr_value1.longs().value(0));
  EXPECT_EQ(2, attr_value1.longs().value(1));
  const auto& broadcast_instr2 = entry_proto.instructions(3);
  ASSERT_EQ(1, broadcast_instr2.attrs().size());
  const auto& attr_value2 =
      broadcast_instr2.attrs().at(note::kBroadcastAlignment);
  ASSERT_TRUE(attr_value2.has_longs());
  EXPECT_EQ(2, attr_value2.longs().value_size());
  EXPECT_EQ(0, attr_value2.longs().value(0));
  EXPECT_EQ(2, attr_value2.longs().value(1));
}

TEST(MetaOpTest, TestUnaryOp) {
  NoteBuilder builder("test_unary_op");
  auto param_op = Parameter(&builder, 0, Shape(note::F32, {1, 2}), "arg");
  auto neg_op = Neg(param_op);
  ASSERT_EQ(&builder, neg_op.Builder());
  ASSERT_TRUE(neg_op.Valid());
  EXPECT_EQ(Shape(note::F32, {1, 2}), neg_op.Shape());
  auto&& module_proto = builder.Build();
  ASSERT_EQ(1, module_proto.functions_size());
  const auto& entry_proto = module_proto.functions(0);
  ASSERT_EQ(2, entry_proto.instructions_size());
  EXPECT_EQ(note::GetOpName(note::OpCode::kParameter),
            entry_proto.instructions(0).opcode());
  EXPECT_EQ(note::GetOpName(note::OpCode::kNegative),
            entry_proto.instructions(1).opcode());
}

TEST(MetaOpTest, TestBinaryOp) {
  NoteBuilder builder("test_binary_op");
  // add broadcast adaptively when shape not equal but compatible
  auto param_op1 = Parameter(&builder, 0, Shape(note::F32, {3}), "arg1");
  auto param_op2 = Parameter(&builder, 1, Shape(note::F32, {2, 3}), "arg2");
  auto add_op = Add(param_op1, param_op2);
  ASSERT_EQ(&builder, add_op.Builder());
  ASSERT_TRUE(add_op.Valid());
  EXPECT_EQ(Shape(note::F32, {2, 3}), add_op.Shape());
  auto&& module_proto = builder.Build();
  const auto& entry_proto = module_proto.functions(0);
  ASSERT_EQ(4, entry_proto.instructions_size());
  EXPECT_EQ(note::GetOpName(note::OpCode::kParameter),
            entry_proto.instructions(0).opcode());
  EXPECT_EQ(note::GetOpName(note::OpCode::kParameter),
            entry_proto.instructions(1).opcode());
  EXPECT_EQ(note::GetOpName(note::OpCode::kBroadcast),
            entry_proto.instructions(2).opcode());
  EXPECT_EQ(note::GetOpName(note::OpCode::kAdd),
            entry_proto.instructions(3).opcode());
}

}  // namespace symbolization
}  // namespace piano
}  // namespace paddle
