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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/llvm_compiler.h"

namespace paddle {
namespace piano {
namespace backends {

KernelExecutableMap LlvmCompiler::Apply(note::Module* note_module) {
  // using pass optimize the note module
  Optimize(note_module);

  // create llvm module
  llvm::LLVMContext context;
  // TODO(sunli) : set llvm_module name.
  llvm::Module llvm_module("", context);

  // create kernel executor
  KernelExecutableMap kernel_executable_map;

  // conver operator to llvm ir
  ConvertToIr(*note_module, &llvm_module, &kernel_executable_map);

  // compiler llvm ir to lowring ir
  Compile(&llvm_module, &kernel_executable_map);

  return kernel_executable_map;
}

void LlvmCompiler::ConvertToIr(const note::Module& note_module,
                               llvm::Module* llvm_module,
                               KernelExecutableMap* kernel_executable_map) {}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
