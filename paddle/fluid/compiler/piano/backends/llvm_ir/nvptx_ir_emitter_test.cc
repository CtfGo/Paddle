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
#include "llvm/IR/Verifier.h"
#include "paddle/fluid/compiler/piano/backends/note_visitor_base.h"
#include "paddle/fluid/compiler/piano/note/function.h"
#include "paddle/fluid/compiler/piano/note/instruction.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/note/opcode.h"
#include "paddle/fluid/compiler/piano/shape.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace piano {
namespace backends {

void CreadInstructionProto(const Shape& shape, const std::string& name,
                           const std::string& op_code, uint64_t id,
                           uint64_t params_index,
                           note::InstructionProto* instr_proto) {
  instr_proto->set_name(name);
  instr_proto->set_opcode(op_code);
  instr_proto->set_id(id);
  instr_proto->set_parameter_index(params_index);
  *instr_proto->mutable_shape() = shape.ToProto();
}

TEST(NvptxIrEmitter, AddOp) {
  const Shape arg1_shape(note::ElementTypeProto::F32, {3, 6});
  const Shape arg2_shape(note::ElementTypeProto::F32, {3, 6});
  const Shape result_shape(note::F32, {3, 6});

  // set arg1_proto
  note::InstructionProto arg1_proto;
  CreadInstructionProto(arg1_shape, "arg1.1", "parameter", 1, 0, &arg1_proto);
  std::unordered_map<std::int64_t, note::Instruction*> instr1_index;
  std::unordered_map<std::int64_t, note::Function*> func_index;
  note::Instruction arg1_instr(arg1_proto, instr1_index, func_index);

  // set arg2_proto
  note::InstructionProto arg2_proto;
  CreadInstructionProto(arg2_shape, "arg2.2", "parameter", 2, 1, &arg2_proto);
  std::unordered_map<std::int64_t, note::Instruction*> instr2_index;
  note::Instruction arg2_instr(arg2_proto, instr2_index, func_index);

  // set add_proto
  note::InstructionProto add_proto;
  CreadInstructionProto(result_shape, "add", "add", 3, 2, &add_proto);
  add_proto.add_operand_ids(1);
  add_proto.add_operand_ids(2);
  std::unordered_map<std::int64_t, note::Instruction*> instr3_index;
  instr3_index.insert(
      std::pair<std::int64_t, note::Instruction*>(1, &arg1_instr));
  instr3_index.insert(
      std::pair<std::int64_t, note::Instruction*>(2, &arg2_instr));
  note::Instruction instr(add_proto, instr3_index, func_index);

  llvm::LLVMContext llvm_context;
  llvm::Module llvm_module("add", llvm_context);
  KernelExecutableMap kernel_executable_map;

  NvptxIrEmitter nvptx_ir_emitter(&llvm_module, &kernel_executable_map);
  nvptx_ir_emitter.VisitAdd(instr);
  llvm_module.print(llvm::errs(), nullptr);

  std::string errors;
  llvm::raw_string_ostream llvm_errors(errors);
  PADDLE_ENFORCE_NE(llvm::verifyModule(llvm_module, &llvm_errors), true,
                    llvm_errors.str());
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
