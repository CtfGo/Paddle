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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_compiler.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Host.h"
#include "paddle/fluid/compiler/piano/note/instruction.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/note/opcode.h"
#include "paddle/fluid/compiler/piano/note_builder.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace piano {
namespace backends {

class TestCompiler : public NvptxCompiler {
 public:
  std::string CallCompileToPtx(llvm::Module* llvm_module) {
    return CompileToPtx(llvm_module);
  }

  std::string CallGetLlvmTriple() { return GetLlvmTriple(); }

  std::string CallGetLlvmDataLayout() { return GetLlvmDataLayout(); }
};

TEST(NvptxCompiler, CompileToPtx) {
  TestCompiler test_compiler;
  // llvm::IR
  llvm::LLVMContext context;
  llvm::Module llvm_module("CuAdd_Test", context);

  llvm_module.setTargetTriple(
      llvm::StringRef(test_compiler.CallGetLlvmTriple()));
  llvm_module.setDataLayout(
      llvm::StringRef(test_compiler.CallGetLlvmDataLayout()));

  {
    llvm::SmallVector<llvm::Type*, 4> args;
    args.push_back(llvm::Type::getFloatPtrTy(context, 1));
    args.push_back(llvm::Type::getFloatPtrTy(context, 1));
    args.push_back(llvm::Type::getFloatPtrTy(context, 1));
    // args.push_back(llvm::Type::getInt32Ty(context));
    llvm::Type* retType = llvm::Type::getVoidTy(context);

    llvm::FunctionType* addType = llvm::FunctionType::get(retType, args, false);
    llvm_module.getOrInsertFunction("CuAdd", addType);
    auto cu_add = llvm_module.getFunction("CuAdd");

    auto args_it = cu_add->arg_begin();
    llvm::Value* arg_a = args_it++;
    llvm::Value* arg_b = args_it++;
    llvm::Value* arg_c = args_it++;
    // llvm::Value* arg_n = args_it ++;

    auto entry = llvm::BasicBlock::Create(context, "entry", cu_add);
    llvm::IRBuilder<> entry_builder(entry);
    llvm::IRBuilder<> add_builder(entry);
    auto tidx = llvm::Intrinsic::getDeclaration(
        &llvm_module,
        llvm::Intrinsic::NVVMIntrinsics::nvvm_read_ptx_sreg_tid_x);
    auto idx = entry_builder.CreateCall(tidx);
    auto _1 = entry_builder.CreateGEP(arg_a, idx);
    auto _2 = entry_builder.CreateGEP(arg_b, idx);
    auto _3 = entry_builder.CreateGEP(arg_c, idx);

    auto _4 = entry_builder.CreateLoad(_1);
    auto _5 = entry_builder.CreateLoad(_2);
    auto _6 = entry_builder.CreateFAdd(_4, _5);
    entry_builder.CreateStore(_6, _3);

    entry_builder.CreateRetVoid();
  }
  llvm_module.print(llvm::errs(), nullptr);
  // log ptx
  LOG(INFO) << test_compiler.CallCompileToPtx(&llvm_module);
}

TEST(NvptxCompiler, Apply) {
  // set device
  platform::SetDeviceId(0);

  // note builder
  NoteBuilder note_builder("test_note_builder");
  {
    note::InstructionProto a_proto, b_proto, c_proto;
    a_proto.set_name("A");
    a_proto.set_opcode("constant");
    auto a_shape = a_proto.mutable_shape();
    a_shape->set_element_type(note::ElementTypeProto::F32);
    a_shape->add_dimensions(1);
    a_shape->add_dimensions(32);

    b_proto.set_name("B");
    b_proto.set_opcode("constant");
    auto b_shape = b_proto.mutable_shape();
    b_shape->set_element_type(note::ElementTypeProto::F32);
    b_shape->add_dimensions(1);
    b_shape->add_dimensions(32);

    c_proto.set_name("C");
    c_proto.set_opcode("add");
    auto c_shape = c_proto.mutable_shape();
    c_shape->set_element_type(note::ElementTypeProto::F32);
    c_shape->add_dimensions(1);
    c_shape->add_dimensions(32);

    // build note module
    std::vector<Operand> ops;
    ops.push_back(note_builder.AppendInstruction(std::move(a_proto),
                                                 note::OpCode::kConstant, {}));
    ops.push_back(note_builder.AppendInstruction(std::move(b_proto),
                                                 note::OpCode::kConstant, {}));
    note_builder.AppendInstruction(std::move(c_proto), note::OpCode::kAdd, ops);
  }
  auto note_proto = note_builder.Build();
  note::Module note_module(note_proto);

  // compile
  NvptxCompiler nvptx_compiler;
  auto kernel_executable_map = nvptx_compiler.Apply(&note_module);
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
