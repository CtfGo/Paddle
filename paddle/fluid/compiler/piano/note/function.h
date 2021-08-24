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
#include "boost/range/iterator_range.hpp"
#include "paddle/fluid/compiler/piano/note/instruction.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/note/type_traits.h"
#include "paddle/fluid/compiler/piano/shape.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace piano {
namespace note {

class Module;

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
  // for(Instruction &instr : function->instructions()){...}
  auto instructions() const {
    using IteratorT = decltype(instructions_.cbegin());
    return boost::make_iterator_range(
        UnboxingIterator<IteratorT>{instructions_.cbegin()},
        UnboxingIterator<IteratorT>{instructions_.cend()});
  }

  // return an instruction included in this function by the given index
  Instruction *instruction(std::int64_t idx) const {
    PADDLE_ENFORCE_EQ(
        idx >= 0 && idx < static_cast<std::int64_t>(instructions_.size()), true,
        platform::errors::PreconditionNotMet("Invalid index value %ld. Its "
                                             "value should between 0(include) "
                                             "and %zu(exclude).",
                                             idx, instructions_.size()));
    PADDLE_ENFORCE_NOT_NULL(
        instructions_[idx].get(),
        platform::errors::PreconditionNotMet(
            "The instruction %ld should not be null.", idx));
    return instructions_[idx].get();
  }

  // return the immutable function signature
  const Signature &signature() const { return signature_; }

  // return the mutable function signature
  Signature *mutable_signature() { return &signature_; }

  // return the globally unique id of this function
  std::int64_t global_id() const { return global_id_; }

  // return the returned instruction of this function
  const Instruction &return_instr() const {
    PADDLE_ENFORCE_NOT_NULL(return_instr_,
                            platform::errors::PreconditionNotMet(
                                "The return instruction should not be null."));
    return *return_instr_;
  }

  // return the immutable module which includes this function
  const Module &parent() const {
    PADDLE_ENFORCE_NOT_NULL(parent_, platform::errors::PreconditionNotMet(
                                         "The parent_(Module) of this function "
                                         "is null, please set it first."));
    return *parent_;
  }

  // return the mutable module which includes this function
  Module *mutable_parent() {
    PADDLE_ENFORCE_NOT_NULL(parent_, platform::errors::PreconditionNotMet(
                                         "The parent_(Module) of this function "
                                         "is null, please set it first."));
    return parent_;
  }

  // set the module in which this function resides
  void set_parent(Module *mod) { parent_ = mod; }

  const std::vector<Instruction *> &param_instrs() const {
    return param_instrs_;
  }

  // return parameter instructions of this function
  const Instruction &param_instr(std::int64_t idx) const {
    PADDLE_ENFORCE_EQ(
        idx >= 0 && idx < static_cast<std::int64_t>(param_instrs_.size()), true,
        platform::errors::PreconditionNotMet("Invalid index value %ld. Its "
                                             "value should between 0(include) "
                                             "and %zu(exclude).",
                                             idx, param_instrs_.size()));
    PADDLE_ENFORCE_NOT_NULL(
        param_instrs_[idx],
        platform::errors::PreconditionNotMet(
            "The parameter instruction %ld should not be null.", idx));
    return *param_instrs_[idx];
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

  // the module where this function is contained
  Module *parent_{nullptr};

  // parameter instructions of this function,
  // which denote input parameters
  std::vector<Instruction *> param_instrs_;

  DISABLE_COPY_AND_ASSIGN(Function);
};

}  // namespace note
}  // namespace piano
}  // namespace paddle
