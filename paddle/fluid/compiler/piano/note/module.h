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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "boost/range/iterator_range.hpp"
#include "paddle/fluid/compiler/piano/note/function.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/note/type_traits.h"
#include "paddle/fluid/compiler/piano/shape.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace piano {
namespace note {

class Module {
 public:
  explicit Module(const ModuleProto& proto);

  ModuleProto ToProto() const;

  std::string ToString() const;

  // return the entry function name
  const std::string& EntryFunctionName() const {
    return entry_function_->name();
  }

  // return the entry function id
  std::int64_t EntryFunctionId() const { return entry_function_->global_id(); }

  // return the module name
  const std::string& name() const { return name_; }

  // return functions included in this module
  // for(Function &instr : module->functions()){...}
  auto functions() const {
    using IteratorT = decltype(functions_.cbegin());
    return boost::make_iterator_range(
        UnboxingIterator<IteratorT>{functions_.cbegin()},
        UnboxingIterator<IteratorT>{functions_.cend()});
  }

  // return a function included in this module by the given index
  Function* function(std::int64_t idx) const {
    PADDLE_ENFORCE_EQ(
        idx >= 0 && idx < static_cast<std::int64_t>(functions_.size()), true,
        platform::errors::PreconditionNotMet("Invalid index value %ld. Its "
                                             "value should between 0(include) "
                                             "and %zu(exclude).",
                                             idx, functions_.size()));
    PADDLE_ENFORCE_NOT_NULL(functions_[idx].get(),
                            platform::errors::PreconditionNotMet(
                                "The function %ld should not be null.", idx));
    return functions_[idx].get();
  }

  // return the immutable entry function signature
  const Signature& entry_signature() const { return entry_signature_; }

  // return the mutable entry function signature
  Signature* mutable_entry_signature() { return &entry_signature_; }

  // return the globally unique id of this module
  std::int64_t global_id() const { return global_id_; }

  // return the immutable entry function of this module
  const Function& entry_function() const {
    PADDLE_ENFORCE_NOT_NULL(entry_function_,
                            platform::errors::PreconditionNotMet(
                                "The entry function of this module "
                                "is null, please set it first."));
    return *entry_function_;
  }

  // return the mutable entry function of this module
  Function* mutable_entry_function() {
    PADDLE_ENFORCE_NOT_NULL(entry_function_,
                            platform::errors::PreconditionNotMet(
                                "The entry function of this module "
                                "is null, please set it first."));
    return entry_function_;
  }

  // set the entry function of this module
  void set_entry_function(Function* func) { entry_function_ = func; }

 private:
  // the module name
  std::string name_;
  // functions included in this module
  std::vector<std::unique_ptr<Function>> functions_;
  // the entry function signature,
  // which includes parameter and return types
  Signature entry_signature_;
  // the global id of this module, which is globally unique
  std::int64_t global_id_;
  // records the entry function
  Function* entry_function_{nullptr};

  DISABLE_COPY_AND_ASSIGN(Module);
};

}  // namespace note
}  // namespace piano
}  // namespace paddle
