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

#include "llvm/IR/IRBuilder.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/primitive_ir_emitter.h"

namespace paddle {
namespace piano {
namespace backends {

using BinaryFunction =
    std::function<llvm::Value*(llvm::Value*, llvm::Value*, llvm::IRBuilder<>*)>;
using UnaryFunction =
    std::function<llvm::Value*(llvm::Value*, llvm::IRBuilder<>*)>;

class GpuPrimitiveIrEmitter : public PrimitiveIrEmitter {
 public:
  GpuPrimitiveIrEmitter(llvm::LLVMContext* ctx, llvm::Function* func)
      : PrimitiveIrEmitter(ctx, func) {}
  ~GpuPrimitiveIrEmitter() {}

  BinaryFunction GetBinaryComputation(const note::Instruction&);
  UnaryFunction GetUnaryComputation(const note::Instruction&);

  void VisitElementwiseUnary(const note::Instruction&) override;
  void VisitElementwiseBinary(const note::Instruction&) override;

  // Scalar op
  void VisitConstant(const note::Instruction&) override;

  // Unary
  void VisitBroadcast(const note::Instruction&) override;
  void VisitCopy(const note::Instruction&) override;
  void VisitReshape(const note::Instruction&) override;
  void VisitReverse(const note::Instruction&) override;
  void VisitSlice(const note::Instruction&) override;
  void VisitTranspose(const note::Instruction&) override;

  // other
  void VisitSelect(const note::Instruction&) override;
  void VisitConcatenate(const note::Instruction&) override;
  void VisitReduce(const note::Instruction&) override;
  void VisitRng(const note::Instruction&) override;
  void VisitSort(const note::Instruction&) override;
  void VisitTuple(const note::Instruction&) override;

  // about the base code block,
  virtual llvm::Value* ThreadIdx(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* ThreadIdy(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* ThreadIdz(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* BlockDimx(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* BlockDimy(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* BlockDimz(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* BlockIdx(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* BlockIdy(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* BlockIdz(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* GridDimx(llvm::IRBuilder<>* ir_builder) = 0;
  virtual llvm::Value* GridDimy(llvm::IRBuilder<>* ir_builder) = 0;
  virtual llvm::Value* GridDimz(llvm::IRBuilder<>* ir_builder) = 0;
  virtual void ThreadSync(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* Alloca(llvm::IRBuilder<>*, unsigned) = 0;
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
