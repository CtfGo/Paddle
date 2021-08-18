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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/primitive_ir_emitter.h"

namespace paddle {
namespace piano {
namespace backends {

void PrimitiveIrEmitter::VisitCast(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

void PrimitiveIrEmitter::VisitExp(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

void PrimitiveIrEmitter::VisitLog(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

void PrimitiveIrEmitter::VisitNegative(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

void PrimitiveIrEmitter::VisitNot(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

void PrimitiveIrEmitter::VisitRsqrt(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

void PrimitiveIrEmitter::VisitSqrt(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

void PrimitiveIrEmitter::VisitAdd(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitAnd(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitCompare(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitDivide(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitMaximum(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitMinimum(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitMultiply(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitOr(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitSubtract(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitXor(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

std::vector<PrimitiveIrGenerator>
PrimitiveIrEmitter::GetPrimitiveIrGenerators() {
  return primitive_ir_generators_;
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
