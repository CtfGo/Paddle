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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/shape.h"

namespace paddle {
namespace piano {
namespace note {

class Instruction;
// class Module;

class Function {
 public:
  // Construct a Function object with a given FunctionProto value.
  // 'func_index' is used to transform function id into Function pointer,
  // which is used to construct instructions in this function.
  Function(const FunctionProto &proto,
           const std::unordered_map<std::int64_t, Function *> &func_index);

  FunctionProto ToProto() const;

  std::string ToString() const;

  // return the name of this function
  const std::string &name() const { return name_; }

  // return instructions owned by this function
  std::vector<Instruction *> instructions() const {
    std::vector<Instruction *> instrs;
    instrs.reserve(instructions_.size());
    std::transform(
        instructions_.cbegin(), instructions_.cend(),
        std::back_inserter(instrs),
        [](const std::unique_ptr<Instruction> &instr) { return instr.get(); });
    return instrs;
  }

  const Instruction *instruction(std::int64_t idx) const {
    return instructions_.at(idx).get();
  }

  Instruction *mutable_instruction(std::int64_t idx) {
    return instructions_.at(idx).get();
  }

  // return the function signature
  const Signature &signature() const { return signature_; }

  Signature *mutable_signature() { return &signature_; }

  std::int64_t global_id() const { return global_id_; }

  // return the returned instruction of this function
  const Instruction *return_instr() const { return return_instr_; }

  // const Module *parent() const { return parent_; }

  // Module *mutable_parent() { return parent_; }

  // void set_parent(Module *module) { parent_ = module; }

  const std::vector<Instruction *> &param_instrs() const {
    return param_instrs_;
  }

  // return parameter instructions of this function
  const Instruction *param_instr(std::int64_t idx) const {
    return param_instrs_.at(idx);
  }

  // return the parameter(input) number of this function
  std::size_t params_num() const { return param_instrs_.size(); }

 private:
  // the name of this function
  std::string name_;
  // instructions owned by this function
  std::vector<std::unique_ptr<Instruction>> instructions_;
  // the function signature, including parameter and return types
  Signature signature_;
  // the global id of this function in a module
  std::int64_t global_id_;
  // the returned instruction of this function
  Instruction *return_instr_;

  // TODO(wzzju): Add Module class.
  // the module where this function is contained
  // Module *parent_{nullptr};

  // parameter instructions of this function,
  // which denote input parameters
  std::vector<Instruction *> param_instrs_;

  DISABLE_COPY_AND_ASSIGN(Function);
};

}  // namespace note
}  // namespace piano
}  // namespace paddle
