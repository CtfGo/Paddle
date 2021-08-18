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

namespace paddle {
namespace piano {
namespace backends {

void GpuPrimitiveIrEmitter::VisitElementwiseUnary(
    const note::Instruction& instr) {}

void GpuPrimitiveIrEmitter::VisitElementwiseBinary(
    const note::Instruction& instr) {}

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
