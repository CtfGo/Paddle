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

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/compiler/piano/backends/note_visitor_base.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/note/opcode.h"
#include "paddle/fluid/compiler/piano/note/type_traits.h"
#include "paddle/fluid/compiler/piano/shape.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace piano {
namespace note {

class Function;

class Instruction {
 public:
  // Construct a Instruction object with a given InstructionProto value.
  // 'instr_index' is used to transform instruction id into Instruction pointer,
  // which is used to fill the operands_ field of this instruction.
  // 'func_index' is used to transform function id into Function pointer,
  // which is used to fill the call_functions_ field of this instruction.
  Instruction(
      const InstructionProto &proto,
      const std::unordered_map<std::int64_t, Instruction *> &instr_index,
      const std::unordered_map<std::int64_t, Function *> &func_index);

  InstructionProto ToProto() const;

  std::string ToString() const;

  void Accept(backends::NoteVisitorBase *visitor);

  void Accept(backends::NoteVisitorBase *visitor) const {
    return const_cast<Instruction *>(this)->Accept(visitor);
  }

  // return the name of this instruction
  const std::string &name() const { return name_; }

  // return the opcode of this instruction
  OpCode opcode() const { return opcode_; }

  const Shape &shape() const { return shape_; }

  Shape *mutable_shape() { return &shape_; }

  const Function *parent() const { return parent_; }

  Function *mutable_parent() { return parent_; }

  void set_parent(Function *func) { parent_ = func; }

  std::int64_t global_id() const { return global_id_; }

  // return instruction operands
  const std::vector<Instruction *> &operands() const { return operands_; }

  const Instruction *operand(std::int64_t idx) const {
    return operands_.at(idx);
  }

  Instruction *mutable_operand(std::int64_t idx) const {
    PADDLE_ENFORCE_NOT_NULL(operands_[idx],
                            platform::errors::PreconditionNotMet(
                                "operand %ld should not be null.", idx));
    return operands_.at(idx);
  }

  const std::vector<Instruction *> &ctrl_predecessors() const {
    return ctrl_predecessors_;
  }

  const std::vector<Instruction *> &ctrl_successors() const {
    return ctrl_successors_;
  }

  // return functions called by this instruction
  const std::vector<Function *> &call_functions() const {
    return call_functions_;
  }

  std::int64_t parameter_number() const { return parameter_number_; }

  // return attributes of this instruction
  const MapType &attrs() const { return attrs_; }

  // get the attribute value according to the given name
  template <typename T>
  T GetAttr(const std::string &attr_name) const {
    PADDLE_ENFORCE_NE(
        attrs_.count(attr_name), 0,
        platform::errors::PreconditionNotMet(
            "%s attribute not in current instruction.", attr_name));
    try {
      return boost::get<T>(attrs_.at(attr_name));
    } catch (boost::bad_get &) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Invalid attribute type of %s, expected: %s, received: %s.",
          attr_name, platform::demangle(typeid(T).name()),
          platform::demangle(attrs_.at(attr_name).type().name())));
    }
  }

 private:
  // get the cpp type value from an attribute proto, which is used to
  // parse a protobuf file
  AttrType GetAttrValue(const AttrValueProto &value_proto) const;

  // get an attribute proto value by the given attribute name, which is
  // used to create a protobuf file
  AttrValueProto GetAttrProto(const std::string &attr_name) const;

  // the name of this instruction
  std::string name_;
  // the opcode of this instruction
  OpCode opcode_;
  // the result shape of this instruction
  Shape shape_;
  // the global id of this instruction in a module
  std::int64_t global_id_;
  // operands of this instruction
  std::vector<Instruction *> operands_;
  // the control predecessors of this instruction
  std::vector<Instruction *> ctrl_predecessors_;
  // the control successors of this instruction
  std::vector<Instruction *> ctrl_successors_;
  // functions called directly by this instruction
  std::vector<Function *> call_functions_;
  // the parameter number of this instruction
  std::int64_t parameter_number_;
  // attributes belongs to this instruction
  MapType attrs_;
  // the function where this instruction is contained
  Function *parent_{nullptr};

  DISABLE_COPY_AND_ASSIGN(Instruction);
};

}  // namespace note
}  // namespace piano
}  // namespace paddle
