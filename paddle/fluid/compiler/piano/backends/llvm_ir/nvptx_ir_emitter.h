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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/gpu_ir_emitter.h"

namespace paddle {
namespace piano {
namespace backends {

// NvptxIrEmitter is used for note::Instruction's implementation with cudnn and
// cublas.

class NvptxIrEmitter : public GpuIrEmitter {
 public:
  NvptxIrEmitter() = delete;
  explicit NvptxIrEmitter(llvm::Module* llvm_module, Schedules* schedule)
      : GpuIrEmitter(llvm_module, schedule) {}
  ~NvptxIrEmitter() {}

  void VisitBatchNormGrad(const note::Instruction&) override;
  void VisitBatchNormInference(const note::Instruction&) override;
  void VisitBatchNormTraining(const note::Instruction&) override;
  void VisitConvolution(const note::Instruction&) override;
  void VisitDot(const note::Instruction&) override;
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
