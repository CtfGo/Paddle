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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/gpu_ir_emitter.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/llvm_utils.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_primitive_ir_emitter.h"
#include "paddle/fluid/compiler/piano/note/instruction.h"

namespace paddle {
namespace piano {
namespace backends {

void GpuIrEmitter::VisitElementwiseUnary(const note::Instruction& instr) {}

void GpuIrEmitter::VisitElementwiseBinary(const note::Instruction& instr) {
  // Step 1: Create kernel function, for binary op, it has 4 args,
  // e.g., add(float*, float*, float*, int)
  auto lhs_type = instr.operand(0).shape().element_type();
  auto rhs_type = instr.operand(1).shape().element_type();
  auto out_type = instr.shape().element_type();
  auto func = CreateLLVMFunction(instr.name(), {lhs_type, rhs_type, out_type},
                                 llvm_module_);

  // Step 2: Create a BasicBlock "entry"
  auto& context = llvm_module_->getContext();
  auto entry_block = llvm::BasicBlock::Create(context, "entry", func);
  llvm::IRBuilder<> entry_irbuilder(entry_block);

  auto args_it = func->arg_begin();
  llvm::Value* lhs = args_it++;
  llvm::Value* rhs = args_it++;
  llvm::Value* out = args_it++;
  llvm::Value* num = args_it++;

  std::unique_ptr<GpuPrimitiveIrEmitter> prim =
      std::make_unique<NvptxPrimitiveIrEmitter>(&context, func);

  // Get thread id and this part will be added to BasicBlock "entry"
  auto tidx = prim->ThreadIdx(&entry_irbuilder);
  auto bidx = prim->BlockIdx(&entry_irbuilder);
  auto block_dim = prim->BlockDimx(&entry_irbuilder);
  auto grid_dim = prim->GridDimx(&entry_irbuilder);
  auto stride = prim->Multiply(block_dim, grid_dim, &entry_irbuilder);
  auto index = prim->Add(prim->Multiply(bidx, block_dim, &entry_irbuilder),
                         tidx, &entry_irbuilder);

  // Step 3: Get the body_generators for the function. There are 3
  // generators "load", "compute" and "store".
  prim->VisitElementwiseBinary(instr);
  auto body_generators = prim->GetPrimitiveIrGenerators();
  auto gen_body = [&](llvm::IRBuilder<>* builder) {
    auto vals = body_generators[0].Run(IrArray{lhs, rhs, index}, builder);
    auto res = body_generators[1].Run(vals, builder);
    body_generators[2].Run(IrArray{res[0], out, index}, builder);
  };

  // Step 4: Combine the for loop and the computation to form the function.
  // PrimitiveIrEmitter::For create 3 BasicBlocks, for_begin, for_body and
  // for_end. The gen_body generates ir in for_body. The function returns
  // void in for_end.
  prim->For(index, num, stride, gen_body, &entry_irbuilder);
  entry_irbuilder.CreateRetVoid();

  // Step 5: Marking the function as kernel
  llvm::NamedMDNode* nvvm_annotations_node =
      llvm_module_->getOrInsertNamedMetadata("nvvm.annotations");
  nvvm_annotations_node->addOperand(llvm::MDNode::get(
      context, {llvm::ConstantAsMetadata::get(func),
                llvm::MDString::get(context, "kernel"),
                llvm::ConstantAsMetadata::get(entry_irbuilder.getInt32(1))}));
}

// Scalar op
void GpuIrEmitter::VisitConstant(const note::Instruction& instr) {}

// Unary
void GpuIrEmitter::VisitBroadcast(const note::Instruction& instr) {}
void GpuIrEmitter::VisitCopy(const note::Instruction& instr) {}
void GpuIrEmitter::VisitReshape(const note::Instruction& instr) {}
void GpuIrEmitter::VisitReverse(const note::Instruction& instr) {}
void GpuIrEmitter::VisitSlice(const note::Instruction& instr) {}
void GpuIrEmitter::VisitTranspose(const note::Instruction& instr) {}

// Other
void GpuIrEmitter::VisitSelect(const note::Instruction& instr) {}
void GpuIrEmitter::VisitConcatenate(const note::Instruction& instr) {}
void GpuIrEmitter::VisitReduce(const note::Instruction& instr) {}
void GpuIrEmitter::VisitRng(const note::Instruction& instr) {}
void GpuIrEmitter::VisitSort(const note::Instruction& instr) {}
void GpuIrEmitter::VisitTuple(const note::Instruction& instr) {}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
