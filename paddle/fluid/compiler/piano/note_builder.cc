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
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace piano {

std::string NameConcatId(const std::string& name, int64_t id,
                         char delim = '.') {
  std::vector<std::string> strs({name, std::to_string(id)});
  return paddle::string::join_strings(strs, delim);
}

const Operand::ShapeType& Operand::Shape() {
  return Builder()->GetShape(*this);
}

Operand NoteBuilder::AppendInstruction(note::InstructionProto&& instr,
                                       note::OpCode opcode,
                                       const std::vector<Operand>& operands) {
  // check the precondition of sevel special instructions
  if (opcode == note::OpCode::kParameter) {
    PADDLE_ENFORCE_EQ(
        instr.has_parameter_number(), true,
        platform::errors::PreconditionNotMet(
            "Parameter instruction shoule fill parameter_number field to "
            "indicate which parameter to be retrieved."));

    const auto& index = instr.parameter_number();
    PADDLE_ENFORCE_EQ(parameter_numbers_.count(index), 0,
                      platform::errors::AlreadyExists(
                          "Parameter[%d] already registered", index));
    parameter_numbers_.insert(index);
  }

  instr.set_id(GetNextId());
  instr.set_opcode(GetOpName(opcode));
  if (instr.name().empty()) {
    instr.set_name(instr.opcode());
  }

  for (const auto& op : operands) {
    PADDLE_ENFORCE_NOT_NULL(
        op.Builder(),
        platform::errors::InvalidArgument(
            "Invalid Operand[%d] because its builder is nullptr", op.Id()));
    PADDLE_ENFORCE_EQ(op.Builder(), this,
                      platform::errors::InvalidArgument(
                          "Operand builder_[%s] not consistent with the "
                          "one[%s] of this instruction",
                          op.Builder()->name(), this->name()));
    instr.add_operand_ids(op.Id());
  }

  id2index_[instr.id()] = instructions_.size();
  instructions_.emplace_back(std::move(instr));
  instruction_shapes_.emplace_back(Shape(instructions_.back().shape()));
  return {instructions_.back().id(), this};
}

const Shape& NoteBuilder::GetShape(Operand op) const {
  PADDLE_ENFORCE_EQ(op.Builder(), this,
                    platform::errors::InvalidArgument(
                        "Operand[%d] not belongs to this builder", op.Id()));
  PADDLE_ENFORCE_GT(id2index_.count(op.Id()), 0,
                    platform::errors::NotFound(
                        "Not found Operand[%d] on this builder", op.Id()));
  return instruction_shapes_[id2index_.at(op.Id())];
}

Signature NoteBuilder::BuildSignature() const {
  Signature signature;
  // by default, the last instruction is root
  *signature.mutable_result() = Shape(instructions_.back().shape());

  signature.mutable_parameters()->resize(parameter_numbers_.size());
  signature.mutable_parameter_names()->resize(parameter_numbers_.size());
  for (const auto& instr : instructions_) {
    static const auto parameter_opcode_name =
        note::GetOpName(note::OpCode::kParameter);
    if (instr.opcode() == parameter_opcode_name) {
      const auto& index = instr.parameter_number();
      // this enforce will ensure the retrieved indexes of kParameter
      // are continuous from 0 to the size;
      PADDLE_ENFORCE_EQ(
          index >= 0 && index < parameter_numbers_.size(), true,
          platform::errors::OutOfRange("parameter number not in range[0, %lld]",
                                       parameter_numbers_.size()));

      signature.mutable_parameters()->at(index) = Shape(instr.shape());
      signature.mutable_parameter_names()->at(index) =
          NameConcatId(instr.name(), instr.id());
    }
  }

  return signature;
}

note::ModuleProto NoteBuilder::Build() {
  PADDLE_ENFORCE_NE(instructions_.empty(), true,
                    platform::errors::PreconditionNotMet(
                        "Can not build note::ModuleProto without instruction"));

  note::FunctionProto entry_function;
  entry_function.set_id(GetNextId());
  entry_function.set_name(NameConcatId(this->name(), entry_function.id()));
  // by default, the last instruction is root
  entry_function.set_return_id(instructions_.back().id());
  *entry_function.mutable_signature() = BuildSignature().ToProto();
  for (auto& instruction : instructions_) {
    instruction.set_name(NameConcatId(instruction.name(), instruction.id()));
    // after building done all data will be cleared,
    // so just take the origin instruction here.
    entry_function.add_instructions()->Swap(&instruction);
  }

  note::ModuleProto note_module;
  note_module.set_id(entry_function.id());
  note_module.set_entry_function_id(entry_function.id());
  note_module.set_name(entry_function.name());
  note_module.set_entry_function_name(entry_function.name());
  note_module.mutable_entry_function_signature()->CopyFrom(
      entry_function.signature());
  // take the origin entry_function directly
  note_module.add_functions()->Swap(&entry_function);

  // Clear data held by this builder.
  this->instructions_.clear();
  this->instruction_shapes_.clear();
  this->id2index_.clear();
  this->parameter_numbers_.clear();

  return note_module;
}

}  // namespace piano
}  // namespace paddle
