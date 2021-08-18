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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/ir_emitter.h"

namespace paddle {
namespace piano {
namespace backends {

// GpuIrEmitter is for translating note::Instruction to llvm IR
// for GPU.
// It implement the visit function for note::Instruction's translation。

class GpuIrEmitter : public IrEmitter {
 public:
  GpuIrEmitter() = delete;
  explicit GpuIrEmitter(llvm::Module* llvm_module, Schedules* schedule)
      : IrEmitter(llvm_module, schedule) {}
  virtual ~GpuIrEmitter() {}

  void VisitElementwiseUnary(const note::Instruction&) override;
  void VisitElementwiseBinary(const note::Instruction&) override;

  // Scalar op
  void VisitConstant(const note::Instruction&) override;

  // ops can be replaced by library
  virtual void VisitBatchNormGrad(const note::Instruction&) = 0;
  virtual void VisitBatchNormInference(const note::Instruction&) = 0;
  virtual void VisitBatchNormTraining(const note::Instruction&) = 0;
  virtual void VisitConvolution(const note::Instruction&) = 0;
  virtual void VisitDot(const note::Instruction&) = 0;

  // Unary
  void VisitBroadcast(const note::Instruction&) override;
  void VisitCopy(const note::Instruction&) override;
  void VisitReshape(const note::Instruction&) override;
  void VisitReverse(const note::Instruction&) override;
  void VisitSlice(const note::Instruction&) override;
  void VisitTranspose(const note::Instruction&) override;

  // Other
  void VisitSelect(const note::Instruction&) override;
  void VisitConcatenate(const note::Instruction&) override;
  void VisitReduce(const note::Instruction&) override;
  void VisitRng(const note::Instruction&) override;
  void VisitSort(const note::Instruction&) override;
  void VisitTuple(const note::Instruction&) override;
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
