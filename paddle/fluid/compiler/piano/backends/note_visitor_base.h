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

namespace paddle {
namespace piano {

namespace note {
class Instruction;
}  // namespace note

namespace backends {

class NoteVisitorBase {
 public:
  virtual ~NoteVisitorBase() {}

  // Scalar op
  virtual void VisitConstant(const note::Instruction&) = 0;
  // TODO(sunli): use the pure virtual function instead
  virtual void VisitParameter(const note::Instruction&) {}

  // ops can be replaced by library
  virtual void VisitBatchNormGrad(const note::Instruction&) = 0;
  virtual void VisitBatchNormInference(const note::Instruction&) = 0;
  virtual void VisitBatchNormTraining(const note::Instruction&) = 0;
  virtual void VisitConvolution(const note::Instruction&) = 0;
  virtual void VisitDot(const note::Instruction&) = 0;

  // Unary
  virtual void VisitBroadcast(const note::Instruction&) = 0;
  virtual void VisitCast(const note::Instruction&) = 0;
  virtual void VisitCopy(const note::Instruction&) = 0;
  virtual void VisitExp(const note::Instruction&) = 0;
  virtual void VisitLog(const note::Instruction&) = 0;
  virtual void VisitNegative(const note::Instruction&) = 0;
  virtual void VisitNot(const note::Instruction&) = 0;
  virtual void VisitReshape(const note::Instruction&) = 0;
  virtual void VisitReverse(const note::Instruction&) = 0;
  virtual void VisitRsqrt(const note::Instruction&) = 0;
  virtual void VisitSlice(const note::Instruction&) = 0;
  virtual void VisitSqrt(const note::Instruction&) = 0;
  virtual void VisitTranspose(const note::Instruction&) = 0;

  // Binary
  virtual void VisitAdd(const note::Instruction&) = 0;
  virtual void VisitAnd(const note::Instruction&) = 0;
  virtual void VisitCompare(const note::Instruction&) = 0;
  virtual void VisitDivide(const note::Instruction&) = 0;
  virtual void VisitMaximum(const note::Instruction&) = 0;
  virtual void VisitMinimum(const note::Instruction&) = 0;
  virtual void VisitMultiply(const note::Instruction&) = 0;
  virtual void VisitOr(const note::Instruction&) = 0;
  virtual void VisitSubtract(const note::Instruction&) = 0;
  virtual void VisitXor(const note::Instruction&) = 0;

  // other
  virtual void VisitSelect(const note::Instruction&) = 0;
  virtual void VisitConcatenate(const note::Instruction&) = 0;
  virtual void VisitReduce(const note::Instruction&) = 0;
  virtual void VisitRng(const note::Instruction&) = 0;
  virtual void VisitSort(const note::Instruction&) = 0;
  virtual void VisitTuple(const note::Instruction&) = 0;
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
