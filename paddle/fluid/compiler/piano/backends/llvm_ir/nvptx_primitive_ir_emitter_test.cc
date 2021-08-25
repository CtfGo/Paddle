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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_primitive_ir_emitter.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace paddle {
namespace piano {
namespace backends {

TEST(NvptxPrimitiveIrEmitter, DeviceBaseOp) {
  llvm::LLVMContext context;
  llvm::Module module("DeviceBaseOp", context);
  llvm::IRBuilder<> builder(context);
  llvm::FunctionType *func_type =
      llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
  llvm::Function *init_fn = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, "init", module);
  llvm::BasicBlock *entry = llvm::BasicBlock::Create(context, "entry", init_fn);
  builder.SetInsertPoint(entry);

  NvptxPrimitiveIrEmitter nvptx_primitive_ir_emitter(&context, init_fn);
  ASSERT_NE(nvptx_primitive_ir_emitter.ThreadIdx(&builder), nullptr);
  ASSERT_NE(nvptx_primitive_ir_emitter.ThreadIdy(&builder), nullptr);
  ASSERT_NE(nvptx_primitive_ir_emitter.ThreadIdz(&builder), nullptr);
  ASSERT_NE(nvptx_primitive_ir_emitter.BlockDimx(&builder), nullptr);
  ASSERT_NE(nvptx_primitive_ir_emitter.BlockDimy(&builder), nullptr);
  ASSERT_NE(nvptx_primitive_ir_emitter.BlockDimz(&builder), nullptr);
  ASSERT_NE(nvptx_primitive_ir_emitter.BlockIdx(&builder), nullptr);
  ASSERT_NE(nvptx_primitive_ir_emitter.BlockIdy(&builder), nullptr);
  ASSERT_NE(nvptx_primitive_ir_emitter.BlockIdz(&builder), nullptr);
  ASSERT_EQ(nvptx_primitive_ir_emitter.Alloca(&builder, 0), nullptr);
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
