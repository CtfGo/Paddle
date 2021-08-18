// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <functional>
#include <string>
#include <vector>
#include "llvm/IR/IRBuilder.h"

namespace paddle {
namespace piano {
namespace backends {

using IrArray = std::vector<llvm::Value*>;
using Generator =
    std::function<IrArray(const IrArray&, llvm::IRBuilder<>* llvm_builder)>;

class PrimitiveIrGenerator {
 public:
  PrimitiveIrGenerator(std::string name, std::string type, Generator generator)
      : generator_name_(name), generator_type_(type), generator_(generator) {}

  ~PrimitiveIrGenerator() {}

  std::string GetName() const { return generator_name_; }

  std::string GetType() const { return generator_type_; }

  Generator& GetGenerator() { return generator_; }

  IrArray Run(const IrArray& ir_array, llvm::IRBuilder<>* ir_builder) {
    return generator_(ir_array, ir_builder);
  }

 private:
  std::string generator_name_;
  std::string generator_type_;
  Generator generator_;
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
