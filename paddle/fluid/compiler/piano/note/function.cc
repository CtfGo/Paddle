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

#include "paddle/fluid/compiler/piano/note/function.h"
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/compiler/piano/note/instruction.h"
#include "paddle/fluid/compiler/piano/note/opcode.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace piano {
namespace note {

Function::Function(
    const FunctionProto& proto,
    const std::unordered_map<std::int64_t, Function*>& func_index)
    : name_(proto.name()),
      signature_(proto.signature()),
      global_id_(proto.id()) {
  // the map used to record `id -> Instruction*`
  std::unordered_map<std::int64_t, Instruction*> instr_index;

  // the map used to record `Instruction* -> id`, which is opposite
  // to the instr_index map
  std::unordered_map<Instruction*, std::int64_t> inverted_index;
  for (const auto& instr_proto : proto.instructions()) {
    auto instr =
        std::make_unique<Instruction>(instr_proto, instr_index, func_index);
    instr->set_parent(this);
    // set parameter(input) instructions field
    if (instr->opcode() == OpCode::kParameter) {
      param_instrs_.push_back(instr.get());
    }
    instr_index[instr_proto.id()] = instr.get();
    inverted_index[instr.get()] = instr_proto.id();
    instructions_.emplace_back(std::move(instr));
  }
  PADDLE_ENFORCE_EQ(
      proto.return_id() >= 0 && instr_index.count(proto.return_id()), true,
      platform::errors::PreconditionNotMet(
          "The return instruction id is %ld, and it is not "
          "included in this function.",
          proto.return_id()));

  // set the returned instruction field
  return_instr_ = instr_index[proto.return_id()];
  std::sort(instructions_.begin(), instructions_.end(),
            [&inverted_index](const std::unique_ptr<Instruction>& l,
                              const std::unique_ptr<Instruction>& r) {
              return inverted_index[l.get()] < inverted_index[r.get()];
            });
}

FunctionProto Function::ToProto() const {
  FunctionProto proto;
  proto.set_name(name_);
  *proto.mutable_signature() = signature_.ToProto();
  proto.set_id(global_id_);
  proto.set_return_id(return_instr_->global_id());
  // serialize instruction protos
  for (const auto& instr : instructions_) {
    *proto.add_instructions() = instr->ToProto();
  }
  return proto;
}

std::string Function::ToString() const {
  std::ostringstream out_str;
  // get the function name and signature
  out_str << "def %" << name_ << signature_.ToString() << " {\n";
  const std::string tab(2, ' ');

  // get the string value of each instruction
  std::size_t num = instructions_.size();
  for (decltype(instructions_.size()) i = 0; i < num; i++) {
    if (num - 1 == i) {
      out_str << tab << "return ";
    } else {
      out_str << tab;
    }
    out_str << instructions_[i]->ToString() << "\n";
  }
  out_str << "}";
  return out_str.str();
}

}  // namespace note
}  // namespace piano
}  // namespace paddle
