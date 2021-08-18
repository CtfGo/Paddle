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

#include "paddle/fluid/compiler/piano/note_builder.h"
#include <utility>
#include "gtest/gtest.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/note/opcode.h"

namespace paddle {
namespace piano {

TEST(OperandTest, Basic) {
  Operand op;
  ASSERT_FALSE(op.Valid());
  // this operand not constructed by a NoteBuilder, and will throw error
  EXPECT_THROW(op.Builder(), paddle::platform::EnforceNotMet);
}

class NoteBuilderTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    instr_arg1_.set_name("arg1");
    *instr_arg1_.mutable_shape() = Shape(note::F32, {2, 3}).ToProto();
    instr_arg1_.set_parameter_number(0);
    instr_arg2_.set_name("arg2");
    *instr_arg2_.mutable_shape() = Shape(note::S8, {2, 3}).ToProto();
    instr_arg2_.set_parameter_number(1);
    instr_add_.set_name("add");
    *instr_add_.mutable_shape() = Shape(note::F64, {2, 3}).ToProto();

    arg1_ = default_builder_.AppendInstruction(std::move(instr_arg1_),
                                               note::OpCode::kParameter, {});
    arg2_ = default_builder_.AppendInstruction(std::move(instr_arg2_),
                                               note::OpCode::kParameter, {});
    add_ = default_builder_.AppendInstruction(
        std::move(instr_add_), note::OpCode::kAdd, {arg1_, arg2_});
  }

  // instructions related to add operation
  note::InstructionProto instr_arg1_, instr_arg2_, instr_add_;
  Operand arg1_, arg2_, add_;
  // default builder
  NoteBuilder default_builder_{"default"};
};

TEST_F(NoteBuilderTest, AppendInstruction) {
  NoteBuilder builder("test_append_instruction");
  ASSERT_EQ("test_append_instruction", builder.name());
  // throw error when precondition not meet on kParameter
  EXPECT_THROW(builder.AppendInstruction(note::InstructionProto(),
                                         note::OpCode::kParameter, {}),
               paddle::platform::EnforceNotMet);

  // check add instruction correctly through return operand
  ASSERT_TRUE(arg1_.Valid());
  EXPECT_EQ(&default_builder_, arg1_.Builder());
  EXPECT_EQ(Shape(note::F32, {2, 3}), arg1_.Shape());
  ASSERT_TRUE(arg2_.Valid());
  EXPECT_EQ(&default_builder_, arg2_.Builder());
  EXPECT_EQ(Shape(note::S8, {2, 3}), arg2_.Shape());
  ASSERT_TRUE(add_.Valid());
  EXPECT_EQ(&default_builder_, add_.Builder());
  EXPECT_EQ(Shape(note::F64, {2, 3}), add_.Shape());
}

TEST_F(NoteBuilderTest, Build) {
  auto&& module_proto = default_builder_.Build();

  // id and name
  ASSERT_EQ(4, module_proto.id());
  EXPECT_EQ(module_proto.id(), module_proto.entry_function_id());
  ASSERT_EQ("default.4", module_proto.name());
  EXPECT_EQ(module_proto.name(), module_proto.entry_function_name());
  EXPECT_EQ(1, module_proto.functions_size());

  // entry function
  const auto& entry_proto = module_proto.functions(0);
  ASSERT_EQ(4, entry_proto.id());
  ASSERT_EQ("default.4", entry_proto.name());
  ASSERT_EQ(3, entry_proto.return_id());
  ASSERT_EQ(3, entry_proto.instructions_size());

  // instructions
  EXPECT_EQ(1, entry_proto.instructions(0).id());
  EXPECT_EQ(note::GetOpName(note::OpCode::kParameter),
            entry_proto.instructions(0).opcode());
  EXPECT_EQ(Shape(note::F32, {2, 3}),
            Shape(entry_proto.instructions(0).shape()));
  EXPECT_EQ(2, entry_proto.instructions(1).id());
  EXPECT_EQ(note::GetOpName(note::OpCode::kParameter),
            entry_proto.instructions(1).opcode());
  EXPECT_EQ(Shape(note::S8, {2, 3}),
            Shape(entry_proto.instructions(1).shape()));
  EXPECT_EQ(3, entry_proto.instructions(2).id());
  EXPECT_EQ(note::GetOpName(note::OpCode::kAdd),
            entry_proto.instructions(2).opcode());
  EXPECT_EQ(Shape(note::F64, {2, 3}),
            Shape(entry_proto.instructions(2).shape()));

  // signature
  const auto& signature_proto = module_proto.entry_function_signature();
  ASSERT_EQ(2, signature_proto.parameters_size());
  EXPECT_EQ(Shape(note::F32, {2, 3}), Shape(signature_proto.parameters(0)));
  EXPECT_EQ("arg1.1", signature_proto.parameter_names(0));
  EXPECT_EQ(Shape(note::S8, {2, 3}), Shape(signature_proto.parameters(1)));
  EXPECT_EQ("arg2.2", signature_proto.parameter_names(1));
  ASSERT_EQ(Shape(note::F64, {2, 3}), Shape(signature_proto.result()));
}

}  // namespace piano
}  // namespace paddle
