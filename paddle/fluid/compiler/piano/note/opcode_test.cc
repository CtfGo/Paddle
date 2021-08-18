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
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace paddle {
namespace piano {
namespace note {

TEST(Opcode, SingleOp) {
  OpCode op = OpCode::kReduce;
  int param_num = GetOpParamNum(op);
  LOG(INFO) << "Op Name = " << GetOpName(op) << ", Param Num = " << param_num;
  ASSERT_EQ(param_num, kVariadicParamNum);
  OpCode code = GetOpCode("broadcast");
  ASSERT_EQ(code, OpCode::kBroadcast);
}

TEST(Opcode, PrintOpCode) {
  const OpCode codes[] = {
#define HANDLE_CODE(enum_id, op_name, ...) OpCode::k##enum_id,
      OPCODE_HANDLER(HANDLE_CODE)
#undef HANDLE_CODE
  };

  auto a = std::make_unique<int>(10);

  for (auto op : codes) {
    auto&& op_name = GetOpName(op);
    int param_num = GetOpParamNum(op);
    LOG(INFO) << "Op Name = " << op_name << ", Param Num = " << param_num;
  }
}

}  // namespace note
}  // namespace piano
}  // namespace paddle
