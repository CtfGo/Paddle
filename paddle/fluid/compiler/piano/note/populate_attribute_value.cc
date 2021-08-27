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

#include "paddle/fluid/compiler/piano/note/populate_attribute_value.h"

namespace paddle {
namespace piano {
namespace note {

// set field 'b' to pass single string
template <>
void PopulateAttrValueProto<std::string>(const std::string& value,
                                         AttrValueProto* attr_value) {
  attr_value->set_s(value);
}

// set field 'b' to pass single bool
template <>
void PopulateAttrValueProto<bool>(const bool& value,
                                  AttrValueProto* attr_value) {
  attr_value->set_b(value);
}

// set field 'i' to pass single int32
template <>
void PopulateAttrValueProto<int32_t>(const int32_t& value,
                                     AttrValueProto* attr_value) {
  attr_value->set_i(value);
}

// set field 'l' to pass single int64
template <>
void PopulateAttrValueProto<int64_t>(const int64_t& value,
                                     AttrValueProto* attr_value) {
  attr_value->set_l(value);
}

// set field 'f' to pass single float
template <>
void PopulateAttrValueProto<float>(const float& value,
                                   AttrValueProto* attr_value) {
  attr_value->set_f(value);
}

// set field 'd' to pass single double
template <>
void PopulateAttrValueProto<double>(const double& value,
                                    AttrValueProto* attr_value) {
  attr_value->set_d(value);
}

// set field 'strings' to pass 1-D string array
template <>
void PopulateAttrValueProto<std::vector<std::string>>(
    const std::vector<std::string>& value, AttrValueProto* attr_value) {
  attr_value->mutable_strings()->mutable_value()->Reserve(value.size());
  *attr_value->mutable_strings()->mutable_value() = {value.begin(),
                                                     value.end()};
}

// set field 'bools' to pass 1-D bool array
template <>
void PopulateAttrValueProto<std::vector<bool>>(const std::vector<bool>& value,
                                               AttrValueProto* attr_value) {
  attr_value->mutable_bools()->mutable_value()->Reserve(value.size());
  *attr_value->mutable_bools()->mutable_value() = {value.begin(), value.end()};
}

// set field 'ints' to pass 1-D int32 array
template <>
void PopulateAttrValueProto<std::vector<int32_t>>(
    const std::vector<int32_t>& value, AttrValueProto* attr_value) {
  attr_value->mutable_ints()->mutable_value()->Reserve(value.size());
  *attr_value->mutable_ints()->mutable_value() = {value.begin(), value.end()};
}

// set field 'longs' to pass 1-D int64 array
template <>
void PopulateAttrValueProto<std::vector<int64_t>>(
    const std::vector<int64_t>& value, AttrValueProto* attr_value) {
  attr_value->mutable_longs()->mutable_value()->Reserve(value.size());
  *attr_value->mutable_longs()->mutable_value() = {value.begin(), value.end()};
}

// set field 'floats' to pass 1-D int64 array
template <>
void PopulateAttrValueProto<std::vector<float>>(const std::vector<float>& value,
                                                AttrValueProto* attr_value) {
  attr_value->mutable_floats()->mutable_value()->Reserve(value.size());
  *attr_value->mutable_floats()->mutable_value() = {value.begin(), value.end()};
}

// set field 'doubles' to pass 1-D int64 array
template <>
void PopulateAttrValueProto<std::vector<double>>(
    const std::vector<double>& value, AttrValueProto* attr_value) {
  attr_value->mutable_doubles()->mutable_value()->Reserve(value.size());
  *attr_value->mutable_doubles()->mutable_value() = {value.begin(),
                                                     value.end()};
}

}  // namespace note
}  // namespace piano
}  // namespace paddle
