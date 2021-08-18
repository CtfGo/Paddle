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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/ir_emitter.h"

namespace paddle {
namespace piano {
namespace backends {

void IrEmitter::VisitCast(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}
void IrEmitter::VisitExp(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}
void IrEmitter::VisitLog(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}
void IrEmitter::VisitNegative(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}
void IrEmitter::VisitNot(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}
void IrEmitter::VisitRsqrt(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}
void IrEmitter::VisitSqrt(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

void IrEmitter::VisitAdd(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}
void IrEmitter::VisitAnd(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}
void IrEmitter::VisitCompare(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}
void IrEmitter::VisitDivide(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}
void IrEmitter::VisitMaximum(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}
void IrEmitter::VisitMinimum(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}
void IrEmitter::VisitMultiply(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}
void IrEmitter::VisitOr(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}
void IrEmitter::VisitSubtract(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}
void IrEmitter::VisitXor(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
