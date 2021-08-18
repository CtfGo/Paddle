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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_primitive_ir_emitter.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/llvm_utils.h"

namespace paddle {
namespace piano {
namespace backends {

std::function<llvm::Value*(llvm::Value*, llvm::IRBuilder<>*)>
NvptxPrimitiveIrEmitter::GetUnaryOp(const note::Instruction& instr) {
  return [&instr](llvm::Value* Value,
                  llvm::IRBuilder<>* ir_builder) -> llvm::Value* {
    return nullptr;
  };
}

std::function<llvm::Value*(llvm::Value*, llvm::Value*, llvm::IRBuilder<>*)>
NvptxPrimitiveIrEmitter::GetBinaryOp(const note::Instruction& instr) {
  return [&instr](llvm::Value* first, llvm::Value* dst,
                  llvm::IRBuilder<>* ir_builder) -> llvm::Value* {
    return nullptr;
  };
}

llvm::Value* NvptxPrimitiveIrEmitter::ThreadIdx(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::ThreadIdy(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::ThreadIdz(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockDimx(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockDimy(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockDimz(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockIdx(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockIdy(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_y;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockIdz(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_z;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

void NvptxPrimitiveIrEmitter::ThreadSync(llvm::IRBuilder<>* ir_builder) {}

llvm::Value* NvptxPrimitiveIrEmitter::Alloca(llvm::IRBuilder<>* ir_builder,
                                             unsigned) {
  return nullptr;
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
