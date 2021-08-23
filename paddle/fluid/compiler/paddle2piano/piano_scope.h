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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/compiler/piano/note_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace piano {

class PianoScope {
 public:
  PianoScope() : parent_(nullptr) {}
  ~PianoScope() = default;

  PianoScope* NewScope() const {
    kids_.emplace_back(new PianoScope(this));
    return kids_.back().get();
  }

  std::unique_ptr<PianoScope> NewTmpScope() const {
    return std::unique_ptr<PianoScope>(new PianoScope(this));
  }

  const PianoScope* parent() const { return parent_; }

  bool HasKid(const PianoScope* scope) const {
    for (auto& kid : kids_) {
      if (kid.get() == scope) {
        return true;
      }
    }
    return false;
  }

  // return "true" if erase success, else return "false"
  bool EraseKid(PianoScope* scope) const {
    PADDLE_ENFORCE_EQ(
        HasKid(scope), true,
        platform::errors::NotFound("Kid scope %p not founded in scope %p.",
                                   scope, this));
    auto it = kids_.begin();
    for (; it != kids_.end(); ++it) {
      if (it->get() == scope) break;
    }
    kids_.erase(it);
    return true;
  }

  // find whether local scope has the operand
  bool HasLocalOperand(const std::string& name) const {
    return operands_.find(name) != operands_.end();
  }

  // find whether local scope and its ancestor scope has the operand
  bool HasOperand(const std::string& name) const {
    return HasLocalOperand(name) ||
           (parent_ != nullptr && parent_->HasOperand(name));
  }

  // Find the scope or an ancestor scope that contains the given operand name
  const PianoScope* FindScope(const std::string& name) const {
    if (HasLocalOperand(name)) {
      return this;
    }
    return (parent_ != nullptr) ? parent_->FindScope(name) : nullptr;
  }

  // return the operand in local scope if founded.
  Operand GetLocalOperand(const std::string& name) const {
    PADDLE_ENFORCE_EQ(
        HasLocalOperand(name), true,
        platform::errors::NotFound("Operand %s not founded in scope %p",
                                   name.c_str(), this));
    return operands_.at(name);
  }

  // return the operand in local scope or its ancestor scope if founded
  Operand GetOperand(const std::string& name) const {
    if (HasLocalOperand(name)) {
      return operands_.at(name);
    }
    PADDLE_ENFORCE_NE(parent_, nullptr,
                      platform::errors::NotFound(
                          "Operand %s not founded in scope", name.c_str()));
    return parent_->GetOperand(name);
  }

  // insert the operand into local scope
  void SetOperand(const std::string& name, const Operand& op) {
    PADDLE_ENFORCE_EQ(HasOperand(name), false,
                      platform::errors::AlreadyExists(
                          "Operand %s already existed in scope %p.",
                          name.c_str(), FindScope(name)));
    operands_.emplace(name, op);
  }

  std::vector<std::string> LocalOperandNames() const {
    std::vector<std::string> ret;
    for (auto& kv : operands_) {
      ret.emplace_back(kv.first);
    }
    return ret;
  }

 private:
  explicit PianoScope(const PianoScope* parent) : parent_(parent) {}

  DISABLE_COPY_AND_ASSIGN(PianoScope);

  std::unordered_map<std::string, Operand> operands_;

  const PianoScope* parent_;
  mutable std::vector<std::unique_ptr<PianoScope>> kids_;
};

}  // namespace piano
}  // namespace paddle
