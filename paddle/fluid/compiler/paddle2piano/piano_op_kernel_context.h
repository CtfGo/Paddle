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
#include <unordered_set>
#include <vector>

#include "paddle/fluid/compiler/paddle2piano/piano_op_registry.h"
#include "paddle/fluid/compiler/paddle2piano/piano_scope.h"
#include "paddle/fluid/compiler/piano/note_builder.h"
#include "paddle/fluid/framework/op_desc.h"

namespace paddle {
namespace piano {

// The PianoOpKernelContext used to transmit the higher level information to the
// lower level of piano note IR.
// "OpDesc" is the operator information.
// "PianoScope" is an association of a name to operand.
// "builder" is the operand's NoteBuilder.
class PianoOpKernelContext {
 public:
  PianoOpKernelContext(const framework::OpDesc* op_desc, PianoScope* scope,
                       NoteBuilder* builder)
      : op_(op_desc), scope_(scope), builder_(builder) {}

  // cannot returning reference to temporary
  std::string Type() const { return op_->Type(); }

  NoteBuilder* Builder() const { return builder_; }

  bool HasInput(const std::string& name) const {
    return op_->Inputs().find(name) != op_->Inputs().end();
  }

  Operand GetInput(const std::string& name) const;

  // Map the outputs's operand into scope, the operand is created by
  // NoteBuilder, and be careful the output name must existed in op's
  // outputs.
  void SetOutput(const std::string& name, const Operand& op) const;

  const std::unordered_set<note::ElementTypeProto>& DataTypes() const {
    return PianoOpRegistry::PianoOpDataTypes(Type());
  }

  // Check whether the attribute exist in Piano or Paddle
  bool HasAttr(const std::string& name) const;

  // The priority of piano attribute is higher than paddle's, so to the same
  // name attribute, GetAttr will return piano's value rather than paddle's.
  framework::Attribute GetAttr(const std::string& name) const;

  template <typename T>
  const T& GetAttr(const std::string& name) const {
    return BOOST_GET_CONST(T, GetAttr(name));
  }

 private:
  const framework::OpDesc* op_;
  mutable PianoScope* scope_;
  NoteBuilder* builder_;
};

}  // namespace piano
}  // namespace paddle
