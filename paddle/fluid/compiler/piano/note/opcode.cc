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

#include "paddle/fluid/compiler/piano/note/opcode.h"
#include <string>
#include <unordered_map>
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace piano {
namespace note {

OpCode GetOpCode(const std::string& op_name) {
  static std::unordered_map<std::string, OpCode> op_map = {
#define HANDLE_MAP(enum_id, op_name, ...) {op_name, OpCode::k##enum_id},
      OPCODE_HANDLER(HANDLE_MAP)
#undef HANDLE_MAP
  };

  PADDLE_ENFORCE_EQ(
      op_map.count(op_name), 1,
      platform::errors::InvalidArgument(
          "Invalid operator name (op_name = %s) !", op_name.c_str()));

  return op_map[op_name];
}

const std::string& GetOpName(OpCode code) {
  static const std::string op_names[] = {
#define HANDLE_NAME(enum_id, op_name, ...) op_name,
      OPCODE_HANDLER(HANDLE_NAME)
#undef HANDLE_NAME
  };
  PADDLE_ENFORCE_LT(
      code, OpCode::kNumOps,
      platform::errors::InvalidArgument("Invalid operator code (opcode = %u) !",
                                        static_cast<std::uint32_t>(code)));

  return op_names[static_cast<std::uint32_t>(code)];
}

int GetOpParamNum(OpCode code) {
  switch (code) {
#define HANDLE_CASE(enum_id, op_name, param_num, ...) \
  case OpCode::k##enum_id:                            \
    return param_num;
    OPCODE_HANDLER(HANDLE_CASE)
#undef HANDLE_CASE

    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Invalid operator code (opcode = %u) !",
          static_cast<std::uint32_t>(code)));
  }
}

}  // namespace note
}  // namespace piano
}  // namespace paddle
