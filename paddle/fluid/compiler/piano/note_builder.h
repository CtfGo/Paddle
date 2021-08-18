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

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/compiler/piano/note/opcode.h"
#include "paddle/fluid/compiler/piano/shape.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace piano {

class NoteBuilder;

// A Operand is generally constructed by a NoteBuilder, as the returned value of
// an Instruction, and can be used as an operand for succeeding instructions
class Operand {
 public:
  using ShapeType = paddle::piano::Shape;

  Operand() : instr_id_(-1), builder_(nullptr) {}

  // whether this operand valid
  bool Valid() const { return instr_id_ >= 0 && builder_ != nullptr; }

  // the builder that contructs this operand
  NoteBuilder* Builder() const {
    PADDLE_ENFORCE_NOT_NULL(
        builder_, platform::errors::InvalidArgument("Builder is nullptr"));
    return builder_;
  }

  // shape of this operand
  const ShapeType& Shape();

 private:
  // declare the folloing methods as private and only can be used from friend
  // class, to prevent from illegal usage in complie-time
  explicit Operand(NoteBuilder* builder) : instr_id_(-1), builder_(builder) {}
  Operand(int64_t id, NoteBuilder* builder)
      : instr_id_(id), builder_(builder) {}

  int64_t Id() const { return instr_id_; }

  friend class NoteBuilder;

 private:
  // the unique id that denotes which instruction generate this value
  int64_t instr_id_;
  // the builder that holds the instruction
  NoteBuilder* builder_;
};

// A NoteBuilder keeps a list of instructions within the same Note Module,
// and user can append new instructions with one or more operands which come
// from instructions enqueued
//
// This is used as a convenient interface for building up the initial Note
// Module.
class NoteBuilder {
 public:
  explicit NoteBuilder(const std::string& name) : name_(name) {}

  // Append an new instruction
  Operand AppendInstruction(note::InstructionProto&& instr, note::OpCode opcode,
                            const std::vector<Operand>& operands);

  // Returns the shape of the given operand.
  const Shape& GetShape(Operand op) const;

  // Build the init note::ModuleProto with an entry function
  // which includes all instructions
  note::ModuleProto Build();

  // name of this builder
  const std::string& name() { return name_; }

 private:
  // Generate the next sequential id
  int64_t GetNextId() { return ++next_id_; }

  // Build the signature of entry function
  Signature BuildSignature() const;

 private:
  // Name to use for the built note::ModuleProto
  std::string name_;

  // The next sequential ID for every instruction contained within this builer.
  int64_t next_id_ = 0;

  // The instructions list
  std::vector<note::InstructionProto> instructions_;

  // The shape list of appended instructions
  std::vector<Shape> instruction_shapes_;

  // A map from ID to the index in the instructions_ vector where the
  // instruction resides in.
  std::unordered_map<int64_t, int64_t> id2index_;

  // The unique parameter numbers.
  std::unordered_set<int64_t> parameter_numbers_;
};

}  // namespace piano
}  // namespace paddle
