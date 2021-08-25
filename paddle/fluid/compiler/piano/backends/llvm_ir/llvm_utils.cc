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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/llvm_utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace piano {
namespace backends {

llvm::Value* CallToLLVMIntrinsic(llvm::IRBuilder<>* ir_builder,
                                 llvm::Intrinsic::ID llvm_Intrinsic) {
  llvm::Module* llvm_module = ir_builder->GetInsertBlock()->getModule();
  llvm::Function* func =
      llvm::Intrinsic::getDeclaration(llvm_module, llvm_Intrinsic);
  return ir_builder->CreateCall(func);
}

llvm::Type* ElementTypeToLLVMType(note::ElementTypeProto element_type,
                                  const llvm::Module& module) {
  auto& ctx = module.getContext();
  switch (element_type) {
    case note::ElementTypeProto::B1:
      return llvm::Type::getIntNTy(ctx, 1);
    case note::ElementTypeProto::S8:
    case note::ElementTypeProto::U8:
      return llvm::Type::getInt8Ty(ctx);
    case note::ElementTypeProto::S16:
    case note::ElementTypeProto::U16:
      return llvm::Type::getInt16Ty(ctx);
    case note::ElementTypeProto::S32:
    case note::ElementTypeProto::U32:
      return llvm::Type::getInt32Ty(ctx);
    case note::ElementTypeProto::S64:
    case note::ElementTypeProto::U64:
      return llvm::Type::getInt64Ty(ctx);
    case note::ElementTypeProto::F16:
      return llvm::Type::getHalfTy(ctx);
    case note::ElementTypeProto::F32:
      return llvm::Type::getFloatTy(ctx);
    case note::ElementTypeProto::F64:
      return llvm::Type::getDoubleTy(ctx);
    default:
      PADDLE_THROW(platform::errors::InvalidArgument("Invalid element type."));
  }
}

llvm::Function* CreateLLVMFunction(
    const std::string& func_name,
    const std::vector<note::ElementTypeProto>& types, llvm::Module* module) {
  auto& ctx = module->getContext();
  std::vector<llvm::Type*> args_types;
  for (auto type : types) {
    args_types.push_back(ElementTypeToLLVMType(type, *module)->getPointerTo());
  }
  args_types.push_back(llvm::Type::getInt32Ty(ctx));
  llvm::ArrayRef<llvm::Type*> argsRef(args_types);

  llvm::FunctionType* func_type = nullptr;
  func_type =
      llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), argsRef, false);
  module->getOrInsertFunction(func_name, func_type);
  auto func = module->getFunction(func_name);
  return func;
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
