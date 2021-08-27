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

#include "llvm/Target/TargetMachine.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/llvm_compiler.h"

namespace paddle {
namespace piano {
namespace backends {

// NvptxCompiler compile llvm ir to executable binary code for nvidia gpu.

class NvptxCompiler : public LlvmCompiler {
 public:
  NvptxCompiler() = default;
  ~NvptxCompiler() {}

 protected:
  void Optimize(note::Module*) override;
  void Compile(const note::Module&, llvm::Module*) override;

 protected:
  std::unique_ptr<llvm::TargetMachine> GetTargetMachine(llvm::Triple);
  std::string CompileToPtx(llvm::Module*);

  // TargeTripe for nvidia gpu, see
  // https://llvm.org/docs/NVPTXUsage.html#triples
  std::string GetLlvmTriple() const { return "nvptx64-nvidia-cuda"; }

  // DataLayout for nvidia gpu, see
  // https://llvm.org/docs/NVPTXUsage.html#data-layout
  std::string GetLlvmDataLayout() const {
    return "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-"
           "f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";
  }
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
