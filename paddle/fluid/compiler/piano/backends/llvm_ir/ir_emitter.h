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

#include "llvm/IR/Module.h"
#include "paddle/fluid/compiler/piano/backends/kernel_executable.h"
#include "paddle/fluid/compiler/piano/backends/note_visitor_base.h"

namespace paddle {
namespace piano {
namespace backends {

// IrEmitter is an  abstract base class for generatting LLVM IR of each
// note::Instruction
// For a special hardware, a XXXIrEmitter should be implemented inheriting from
// IrEmitter and overwrites all the virtual functions.
// llvm::Module* -> contain kernel llvm IR
// KernelExecutableMap -> a map of KernelExecutable
// XXXIrEmitter get a llvm::Module* and KernelExecutors* from XXXCompiler when
// initialize.
// note::Instruction accept a IrEmitter and choose VisitXXX by note::OpCode
// Each time VisitXXX will translate one note::Instruction with type OpCode::XXX
// into a kernel with llvm IR.

class IrEmitter : public NoteVisitorBase {
 public:
  IrEmitter() = delete;
  explicit IrEmitter(llvm::Module* llvm_module,
                     KernelExecutableMap* kernel_executable_map)
      : llvm_module_(llvm_module),
        kernel_executable_map_(kernel_executable_map) {}
  virtual ~IrEmitter() {}

  // Elementwise-Unary implemented in VisitElementwiseUnary
  virtual void VisitElementwiseUnary(const note::Instruction&) = 0;
  // Elementwise-Unary implemented in VisitElementwiseBinary
  virtual void VisitElementwiseBinary(const note::Instruction&) = 0;

  // Scalar op
  virtual void VisitConstant(const note::Instruction&) = 0;

  // ops can be replaced by library
  virtual void VisitBatchNormGrad(const note::Instruction&) = 0;
  virtual void VisitBatchNormInference(const note::Instruction&) = 0;
  virtual void VisitBatchNormTraining(const note::Instruction&) = 0;
  virtual void VisitConvolution(const note::Instruction&) = 0;
  virtual void VisitDot(const note::Instruction&) = 0;

  // Unary
  virtual void VisitBroadcast(const note::Instruction&) = 0;
  virtual void VisitCopy(const note::Instruction&) = 0;
  virtual void VisitReshape(const note::Instruction&) = 0;
  virtual void VisitReverse(const note::Instruction&) = 0;
  virtual void VisitSlice(const note::Instruction&) = 0;
  virtual void VisitTranspose(const note::Instruction&) = 0;

  // Unary Compute
  void VisitCast(const note::Instruction&) override;
  void VisitExp(const note::Instruction&) override;
  void VisitLog(const note::Instruction&) override;
  void VisitNegative(const note::Instruction&) override;
  void VisitNot(const note::Instruction&) override;
  void VisitRsqrt(const note::Instruction&) override;
  void VisitSqrt(const note::Instruction&) override;

  // Binary
  void VisitAdd(const note::Instruction&) override;
  void VisitAnd(const note::Instruction&) override;
  void VisitCompare(const note::Instruction&) override;
  void VisitDivide(const note::Instruction&) override;
  void VisitMaximum(const note::Instruction&) override;
  void VisitMinimum(const note::Instruction&) override;
  void VisitMultiply(const note::Instruction&) override;
  void VisitOr(const note::Instruction&) override;
  void VisitSubtract(const note::Instruction&) override;
  void VisitXor(const note::Instruction&) override;

  // Other
  virtual void VisitSelect(const note::Instruction&) = 0;
  virtual void VisitConcatenate(const note::Instruction&) = 0;
  virtual void VisitReduce(const note::Instruction&) = 0;
  virtual void VisitRng(const note::Instruction&) = 0;
  virtual void VisitSort(const note::Instruction&) = 0;
  virtual void VisitTuple(const note::Instruction&) = 0;

 protected:
  llvm::Module* llvm_module_{nullptr};
  KernelExecutableMap* kernel_executable_map_{nullptr};
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
