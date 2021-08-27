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

#pragma once

namespace paddle {
namespace piano {
namespace note {

// In this file, we define the attribute key name for
// specific note instructions. Format of name as bellow
//  `k(p0).(p1)`
//  -`p0`: the instruction name
//  -`p1` the specific attribute name

// literal value of Constant instruction
constexpr char kConstantValue[] = "kConstant.Value";
// `dimensions_alignment` of Broadcast instruction
constexpr char kBroadcastAlignment[] = "kBroadcast.Alignment";

}  // namespace note
}  // namespace piano
}  // namespace paddle
