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
  /*
      const int64_t lhs_rank = lhs_shape->rank();
      const int64_t rhs_rank = rhs_shape->rank();

      Operand updated_lhs = lhs;
      Operand updated_rhs = rhs;
      if (!broadcast_dimensions.empty() && lhs_rank != rhs_rank) {
        const bool should_broadcast_lhs = lhs_rank < rhs_rank;
        Operand from = should_broadcast_lhs ? lhs : rhs;
        const Shape& from_shape = should_broadcast_lhs ? *lhs_shape :
     *rhs_shape;

        std::vector<int64> to_size;
        std::vector<bool> to_size_is_dynamic;
        for (int i = 0; i < shape.rank(); i++) {
          to_size.push_back(shape.dimensions(i));
          to_size_is_dynamic.push_back(shape.is_dynamic_dimension(i));
        }
        for (int64_t from_dim = 0; from_dim < from_shape.rank(); from_dim++) {
          int64_t to_dim = broadcast_dimensions[from_dim];
          to_size[to_dim] = from_shape.dimensions(from_dim);
          to_size_is_dynamic[to_dim] =
     from_shape.is_dynamic_dimension(from_dim);
        }

        const Shape& broadcasted_shape = ShapeUtil::MakeShape(
            from_shape.element_type(), to_size, to_size_is_dynamic);
        TF_ASSIGN_OR_RETURN(
            Operand broadcasted_operand,
            InDimBroadcast(broadcasted_shape, from, broadcast_dimensions));

        updated_lhs = should_broadcast_lhs ? broadcasted_operand : lhs;
        updated_rhs = !should_broadcast_lhs ? broadcasted_operand : rhs;
      }

      TF_ASSIGN_OR_RETURN(const Shape* updated_lhs_shape,
                          GetShapePtr(updated_lhs));
      if (!ShapeUtil::SameDimensions(shape, *updated_lhs_shape)) {
        TF_ASSIGN_OR_RETURN(updated_lhs,
                            AddBroadcastSequence(shape, updated_lhs));
      }
      TF_ASSIGN_OR_RETURN(const Shape* updated_rhs_shape,
                          GetShapePtr(updated_rhs));
      if (!ShapeUtil::SameDimensions(shape, *updated_rhs_shape)) {
        TF_ASSIGN_OR_RETURN(updated_rhs,
                            AddBroadcastSequence(shape, updated_rhs));
      }
      */
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
