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

#include "paddle/fluid/compiler/paddle2piano/piano_op_kernel_context.h"

#include <algorithm>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace piano {

bool PianoOpKernelContext::HasAttr(const std::string& name) const {
  const auto& attrs = PianoOpRegistry::Attrs(Type());
  if (attrs.find(name) != attrs.end()) {
    return true;
  }
  return op_->HasAttr(name);
}

framework::Attribute PianoOpKernelContext::GetAttr(
    const std::string& name) const {
  PADDLE_ENFORCE_EQ(
      HasAttr(name), true,
      platform::errors::NotFound("Attribute %s is not found in op %s.",
                                 name.c_str(), op_->Type().c_str()));

  const auto& attrs = PianoOpRegistry::Attrs(Type());
  auto it = attrs.find(name);
  if (it != attrs.end()) {
    return it->second;
  }
  return op_->GetAttr(name);
}

Operand PianoOpKernelContext::GetInput(const std::string& name) const {
  PADDLE_ENFORCE_EQ(
      HasInput(name), true,
      platform::errors::NotFound("Input %s is not found in op %s.",
                                 name.c_str(), op_->Type().c_str()));
  return scope_->GetOperand(name);
}

void PianoOpKernelContext::SetOutput(const std::string& name,
                                     const Operand& op) const {
  PADDLE_ENFORCE_EQ(
      op_->HasOutput(name), true,
      platform::errors::NotFound("Output %s is not found in op %s.",
                                 name.c_str(), op_->Type().c_str()));
  scope_->SetOperand(name, op);
}

}  // namespace piano
}  // namespace paddle
