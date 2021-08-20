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

#include "paddle/fluid/compiler/paddle2piano/piano_op_registry.h"

#include <string>

#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace piano {

void PianoOpRegistry::RegisterBackend(
    const std::string& backend_name,
    const std::unordered_set<note::ElementTypeProto>& supported_types,
    BackendFilterFunc filter_func) {
  PADDLE_ENFORCE_EQ(
      PianoOpRegistry::IsBackend(backend_name), false,
      platform::errors::AlreadyExists("Backend %s has been registered.",
                                      backend_name.c_str()));
  auto& registry = Instance();
  registry.backend_.emplace(backend_name, new Backend);

  auto& backend = registry.backend_.at(backend_name);
  backend->name = backend_name;
  backend->supported_types = supported_types;
  backend->filter_func = filter_func;
}

const std::unordered_set<note::ElementTypeProto>&
PianoOpRegistry::BackendDataTypes(const std::string& backend_name) {
  PADDLE_ENFORCE_EQ(IsBackend(backend_name), true,
                    platform::errors::NotFound("Name %s not founded Backend.",
                                               backend_name.c_str()));
  return Instance().backend_.at(backend_name)->supported_types;
}

std::vector<std::string> PianoOpRegistry::AllBackendNames() {
  auto& registry = Instance();
  std::vector<std::string> ret;
  for (const auto& backend_pair : registry.backend_) {
    ret.emplace_back(backend_pair.first);
  }
  return ret;
}

bool PianoOpRegistry::HasAllowBackendList(const std::string& op_type) {
  PADDLE_ENFORCE_EQ(
      IsPianoOp(op_type), true,
      platform::errors::NotFound("OP %s is not Piano Op.", op_type.c_str()));
  return Instance().ops_.at(op_type)->has_allow_backend_list;
}

std::vector<std::string> PianoOpRegistry::AllPianoOps() {
  auto& registry = Instance();
  std::vector<std::string> ret;
  for (const auto& op_pair : registry.ops_) {
    ret.emplace_back(op_pair.first);
  }
  return ret;
}

const PianoOpRegistry::OpKernelMap& PianoOpRegistry::AllPianoOpKernels(
    const std::string& op_type) {
  PADDLE_ENFORCE_EQ(
      IsPianoOp(op_type), true,
      platform::errors::NotFound("OP %s is not Piano Op.", op_type.c_str()));

  return Instance().ops_.at(op_type)->kernel_;
}

const framework::AttributeMap& PianoOpRegistry::Attrs(
    const std::string& op_type) {
  PADDLE_ENFORCE_EQ(
      PianoOpRegistry::IsPianoOp(op_type), true,
      platform::errors::NotFound("OP %s is not Piano Op.", op_type.c_str()));

  return Instance().ops_.at(op_type)->attrs;
}

const std::unordered_set<note::ElementTypeProto>&
PianoOpRegistry::PianoOpDataTypes(const std::string& op_type) {
  PADDLE_ENFORCE_EQ(
      PianoOpRegistry::IsPianoOp(op_type), true,
      platform::errors::NotFound("OP %s is not Piano Op.", op_type.c_str()));

  return Instance().ops_.at(op_type)->supported_types;
}

}  // namespace piano
}  // namespace paddle
