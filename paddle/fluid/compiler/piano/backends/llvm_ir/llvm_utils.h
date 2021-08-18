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

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"

namespace paddle {
namespace piano {
namespace backends {

llvm::Value* CallToLLVMIntrinsic(llvm::IRBuilder<>* ir_builder,
                                 llvm::Intrinsic::ID llvm_Intrinsic) {
  llvm::Module* llvm_module = ir_builder->GetInsertBlock()->getModule();
  llvm::Function* func =
      llvm::Intrinsic::getDeclaration(llvm_module, llvm_Intrinsic);
  return ir_builder->CreateCall(func);
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
