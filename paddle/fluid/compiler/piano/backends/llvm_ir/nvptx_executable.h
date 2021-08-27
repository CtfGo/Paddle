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

#include "paddle/fluid/compiler/piano/backends/kernel_executable.h"

namespace paddle {
namespace piano {
namespace backends {

// Nvptx Executable for nvidia jit execution.
// TODO(sunli) : overwrite the function Run.
class NvtpxExecutable : public KernelExecutable {
 public:
  NvtpxExecutable(const std::string module_name,
                  const note::Instruction& note_instruction)
      : KernelExecutable(note_instruction) {
    module_name_ = module_name;
  }
  void Launch(const ExecutableContext&) override;

 private:
  int device_to_execute_{-1};
  std::string module_name_;
  CUfunction cu_function_{nullptr};
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
