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

#include "paddle/fluid/compiler/piano/note/module.h"
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include "paddle/fluid/compiler/piano/note/function.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace piano {
namespace note {

Module::Module(const ModuleProto& proto)
    : name_(proto.name()),
      entry_signature_(proto.entry_function_signature()),
      global_id_(proto.id()) {
  // the map used to record `id -> Function*`
  std::unordered_map<std::int64_t, Function*> func_index;

  // the map used to record `Function* -> id`, which is opposite
  // to the func_index map
  std::unordered_map<Function*, std::int64_t> inverted_index;

  // construct functions from proto
  for (const auto& func_proto : proto.functions()) {
    auto func = std::make_unique<Function>(func_proto, func_index);
    func->set_parent(this);
    auto func_id = func_proto.id();
    PADDLE_ENFORCE_EQ(
        func_index.count(func_id), 0,
        platform::errors::PreconditionNotMet(
            "The global id (%ld) of Function %s is the same as the previous "
            "Function %s.",
            func_id, func->name(), func_index[func_id]->name()));
    func_index[func_id] = func.get();
    inverted_index[func.get()] = func_id;
    if (func_id == proto.entry_function_id()) {
      entry_function_ = func.get();
    }
    functions_.emplace_back(std::move(func));
  }

  PADDLE_ENFORCE_NOT_NULL(
      entry_function_,
      platform::errors::PreconditionNotMet(
          "The entry_function_id in Proto is not a valid function id."));

  std::sort(functions_.begin(), functions_.end(),
            [&inverted_index](const std::unique_ptr<Function>& l,
                              const std::unique_ptr<Function>& r) {
              return inverted_index[l.get()] < inverted_index[r.get()];
            });
}

ModuleProto Module::ToProto() const {
  ModuleProto proto;
  proto.set_name(name_);
  // serialize entry function name
  proto.set_entry_function_name(EntryFunctionName());
  *proto.mutable_entry_function_signature() = entry_signature_.ToProto();
  proto.set_id(global_id_);
  // serialize entry function id
  proto.set_entry_function_id(entry_function_->global_id());
  // serialize function protos in this module
  for (const auto& func : functions_) {
    *proto.add_functions() = func->ToProto();
  }

  return proto;
}

std::string Module::ToString() const {
  std::ostringstream out_str;
  out_str << "Module " << name_ << "\n\n";
  for (const auto& func : functions_) {
    if (func->global_id() == entry_function_->global_id()) {
      out_str << "[[entry]] ";
    }
    out_str << func->ToString() << "\n\n";
  }
  return out_str.str();
}

}  // namespace note
}  // namespace piano
}  // namespace paddle
