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

#include "paddle/fluid/compiler/piano/meta_op_util.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/note_builder.h"
#include "paddle/fluid/compiler/piano/shape_inference.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace piano {

Operand UnaryOp(Opcode unop, Operand x) {
  note::InstructionProto instr;
  auto&& shape = InferUnaryOpShape(unop, x.Shape());
  *instr.mutable_shape() = shape.ToProto();
  return x.Builder()->AppendInstruction(std::move(instr), unop, {x});
}

std::pair<Operand, Operand> AddBroadCastForeOp(
    Operand x, Operand y, const std::vector<int64_t>& broadcast_dimensions) {
  return {x, y};
}

Operand BinaryOp(Opcode binop, Operand x, Operand y,
                 const std::vector<int64_t>& broadcast_dimensions) {
  auto&& ret_shape =
      InferBinaryOpShape(binop, x.Shape(), y.Shape(), broadcast_dimensions);

  auto && [ lhs, rhs ] = AddBroadCastForeOp(x, y, broadcast_dimensions);
  note::InstructionProto instr;
  *instr.mutable_shape() = ret_shape.ToProto();
  return x.Builder()->AppendInstruction(std::move(instr), binop, {x, y});
}

}  // namespace piano
}  // namespace paddle
