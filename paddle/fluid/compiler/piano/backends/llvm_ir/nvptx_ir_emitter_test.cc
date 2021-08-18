/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_ir_emitter.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace paddle {
namespace piano {
namespace backends {

TEST(NvptxIrEmitter, TestOp) {
  llvm::LLVMContext llvm_context;
  llvm::Module llvm_module("test", llvm_context);
  Schedules schedules;

  // create ir emitter
  NvptxIrEmitter nvptx_ir_emitter(&llvm_module, &schedules);
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
