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

#include <cstdint>
#include <string>
#include <vector>
#include "paddle/fluid/compiler/piano/note/note.pb.h"

namespace paddle {
namespace piano {
namespace note {

// Populate the corresponding field of AttrValueProto
// according to the type of input value which is a scalar
// or a 1-D array hold by std::vector
template <typename NativeT>
void PopulateAttrValueProto(const NativeT& value, AttrValueProto* attr_value) {
  static_assert(!std::is_same<NativeT, NativeT>::value,
                "This NativeT is not supported");
}

// Declarations of specializations for each supported native type of a scalar
template <>
void PopulateAttrValueProto<std::string>(const std::string& value,
                                         AttrValueProto* attr_value);
template <>
void PopulateAttrValueProto<bool>(const bool& value,
                                  AttrValueProto* attr_value);
template <>
void PopulateAttrValueProto<int32_t>(const int32_t& value,
                                     AttrValueProto* attr_value);
template <>
void PopulateAttrValueProto<int64_t>(const int64_t& value,
                                     AttrValueProto* attr_value);
template <>
void PopulateAttrValueProto<float>(const float& value,
                                   AttrValueProto* attr_value);
template <>
void PopulateAttrValueProto<double>(const double& value,
                                    AttrValueProto* attr_value);

// Declarations of specializations for each supported native type
// of a 1-D array hold by std::vector
template <>
void PopulateAttrValueProto<std::vector<std::string>>(
    const std::vector<std::string>& value, AttrValueProto* attr_value);
template <>
void PopulateAttrValueProto<std::vector<bool>>(const std::vector<bool>& value,
                                               AttrValueProto* attr_value);
template <>
void PopulateAttrValueProto<std::vector<int32_t>>(
    const std::vector<int32_t>& value, AttrValueProto* attr_value);
template <>
void PopulateAttrValueProto<std::vector<int64_t>>(
    const std::vector<int64_t>& value, AttrValueProto* attr_value);
template <>
void PopulateAttrValueProto<std::vector<float>>(const std::vector<float>& value,
                                                AttrValueProto* attr_value);
template <>
void PopulateAttrValueProto<std::vector<double>>(
    const std::vector<double>& value, AttrValueProto* attr_value);

}  // namespace note
}  // namespace piano
}  // namespace paddle
