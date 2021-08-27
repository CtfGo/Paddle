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

#include "paddle/fluid/compiler/piano/symbolization/shape_inference.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace piano {
namespace symbolization {

Shape InferUnaryOpShape(note::OpCode opcode, const Shape& shape) {
  // TODO(CtfGo):
  // add some checks that verify the element_type of shape is supported
  // for specific opcodes
  return shape;
}

Shape InferBinaryOpShape(note::OpCode opcode, const Shape& lhs,
                         const Shape& rhs) {
  // TODO(CtfGo):
  // add some checks that verify the element_type of the two operands
  // are compatible for specific opcodes
  return lhs;
}

Shape InferBroadcastShape(const Shape& operand_shape,
                          const std::vector<int64_t>& out_dimensions,
                          const std::vector<int64_t>& dimensions_alignment) {
  PADDLE_ENFORCE_EQ(operand_shape.IsArray(), true,
                    platform::errors::InvalidArgument(
                        "Shape of operand should be array tuple"));
  PADDLE_ENFORCE_LE(operand_shape.Rank(), out_dimensions.size(),
                    platform::errors::InvalidArgument(
                        "Rank of operand should be less than output"));
  PADDLE_ENFORCE_EQ(
      dimensions_alignment.size(), operand_shape.Rank(),
      platform::errors::InvalidArgument(
          "Rank of operand should be equal to dimensions_alignment size"));

  // check the length of each out dimension are positive
  for (auto&& dim_size : out_dimensions) {
    PADDLE_ENFORCE_GT(
        dim_size, 0,
        platform::errors::OutOfRange(
            "Broadcast with negative dimension size %d.", dim_size));
  }

  // check dimensions_alignment valid
  for (auto i = 0; i < operand_shape.Rank(); ++i) {
    auto&& to_dim = dimensions_alignment.at(i);
    PADDLE_ENFORCE_EQ(to_dim >= 0 && to_dim < out_dimensions.size(), true,
                      platform::errors::OutOfRange(
                          "Invalid broadcast alignment[%d] on dimension:%d,"
                          "alignment must be non-negative and less than the "
                          "out_dimensions size",
                          to_dim, i));
    const auto& ori_dimsize = operand_shape.dimensions().at(i);

    PADDLE_ENFORCE_EQ(
        ori_dimsize == 1 || ori_dimsize == out_dimensions[to_dim], true,
        platform::errors::OutOfRange("Input dimension should be either 1 or "
                                     "equal to the output dimension"));

    // Make sure the broadcast dimensions alignment are
    // listed in a strictly increasing order.
    PADDLE_ENFORCE_EQ(
        i == 0 || dimensions_alignment[i - 1] < dimensions_alignment[i], true,
        platform::errors::InvalidArgument(
            "Broadcast alignment order is wrong: %d comes after %d.",
            dimensions_alignment[i], dimensions_alignment[i - 1]));
  }

  return {operand_shape.element_type(), out_dimensions};
}

}  // namespace symbolization
}  // namespace piano
}  // namespace paddle
