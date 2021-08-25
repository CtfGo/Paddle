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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/gpu_primitive_ir_emitter.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/primitive_ir_emitter.h"
#include "paddle/fluid/compiler/piano/note/instruction.h"
#include "paddle/fluid/compiler/piano/note/opcode.h"

namespace paddle {
namespace piano {
namespace backends {

BinaryFunction GpuPrimitiveIrEmitter::GetBinaryComputation(
    const note::Instruction& instr) {
  return [&instr, this](llvm::Value* lhs, llvm::Value* rhs,
                        llvm::IRBuilder<>* builder) -> llvm::Value* {
    switch (instr.opcode()) {
      case note::OpCode::kAdd:
        return this->Add(lhs, rhs, builder);
      case note::OpCode::kMultiply:
        return this->Multiply(lhs, rhs, builder);
      default:
        PADDLE_THROW(platform::errors::InvalidArgument("Invalid OpCode."));
    }
  };
}

UnaryFunction GpuPrimitiveIrEmitter::GetUnaryComputation(
    const note::Instruction& instr) {
  return nullptr;
}

void GpuPrimitiveIrEmitter::VisitElementwiseUnary(
    const note::Instruction& instr) {}

void GpuPrimitiveIrEmitter::VisitElementwiseBinary(
    const note::Instruction& instr) {
  // Load
  primitive_ir_generators_.emplace_back(
      "Load_" + instr.name(), "LOAD",
      [this](IrArray llvm_values, llvm::IRBuilder<>* llvm_builder) {
        // (lhs, rhs, index) = llvm_values[0:3]
        return IrArray{Load(llvm_values[0], llvm_values[2], llvm_builder),
                       Load(llvm_values[1], llvm_values[2], llvm_builder)};
      });
  // Compute
  primitive_ir_generators_.emplace_back(
      "Compute_" + instr.name(), "COMPUTE",
      [this, &instr](IrArray llvm_values, llvm::IRBuilder<>* llvm_builder) {
        return IrArray{GetBinaryComputation(instr)(
            llvm_values[0], llvm_values[1], llvm_builder)};
      });
  // Store
  primitive_ir_generators_.emplace_back(
      "Store_" + instr.name(), "STORE",
      [this](IrArray llvm_values, llvm::IRBuilder<>* llvm_builder) {
        // (src, dst, dst_index) = llvm_values[0:3]
        return IrArray{Store(llvm_values[0], llvm_values[1], llvm_values[2],
                             llvm_builder)};
      });
}

// Scalar op
void GpuPrimitiveIrEmitter::VisitConstant(const note::Instruction& instr) {}

// Unary
void GpuPrimitiveIrEmitter::VisitBroadcast(const note::Instruction& instr) {}
void GpuPrimitiveIrEmitter::VisitCopy(const note::Instruction& instr) {}
void GpuPrimitiveIrEmitter::VisitReshape(const note::Instruction& instr) {}
void GpuPrimitiveIrEmitter::VisitReverse(const note::Instruction& instr) {}
void GpuPrimitiveIrEmitter::VisitSlice(const note::Instruction& instr) {}
void GpuPrimitiveIrEmitter::VisitTranspose(const note::Instruction& instr) {}

// Other
void GpuPrimitiveIrEmitter::VisitSelect(const note::Instruction& instr) {}
void GpuPrimitiveIrEmitter::VisitConcatenate(const note::Instruction& instr) {}
void GpuPrimitiveIrEmitter::VisitReduce(const note::Instruction& instr) {}
void GpuPrimitiveIrEmitter::VisitRng(const note::Instruction& instr) {}
void GpuPrimitiveIrEmitter::VisitSort(const note::Instruction& instr) {}
void GpuPrimitiveIrEmitter::VisitTuple(const note::Instruction& instr) {}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
