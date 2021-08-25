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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/gpu_primitive_ir_emitter.h"

namespace paddle {
namespace piano {
namespace backends {

class NvptxPrimitiveIrEmitter : public GpuPrimitiveIrEmitter {
 public:
  NvptxPrimitiveIrEmitter(llvm::LLVMContext* ctx, llvm::Function* func)
      : GpuPrimitiveIrEmitter(ctx, func) {}
  ~NvptxPrimitiveIrEmitter() {}

  // block size
  llvm::Value* ThreadIdx(llvm::IRBuilder<>*) override;
  llvm::Value* ThreadIdy(llvm::IRBuilder<>*) override;
  llvm::Value* ThreadIdz(llvm::IRBuilder<>*) override;
  llvm::Value* BlockDimx(llvm::IRBuilder<>*) override;
  llvm::Value* BlockDimy(llvm::IRBuilder<>*) override;
  llvm::Value* BlockDimz(llvm::IRBuilder<>*) override;
  llvm::Value* BlockIdx(llvm::IRBuilder<>*) override;
  llvm::Value* BlockIdy(llvm::IRBuilder<>*) override;
  llvm::Value* BlockIdz(llvm::IRBuilder<>*) override;
  llvm::Value* GridDimx(llvm::IRBuilder<>*) override;
  llvm::Value* GridDimy(llvm::IRBuilder<>*) override;
  llvm::Value* GridDimz(llvm::IRBuilder<>*) override;
  void ThreadSync(llvm::IRBuilder<>*) override;
  llvm::Value* Alloca(llvm::IRBuilder<>*, unsigned) override;
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
