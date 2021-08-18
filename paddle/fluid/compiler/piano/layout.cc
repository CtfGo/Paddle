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

#include "paddle/fluid/compiler/piano/layout.h"
#include <algorithm>
#include <iterator>
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace piano {

Layout::Layout(const note::LayoutProto& proto) {
  minor_to_major_.reserve(proto.minor_to_major_size());
  minor_to_major_.insert(minor_to_major_.end(), proto.minor_to_major().begin(),
                         proto.minor_to_major().end());
}

note::LayoutProto Layout::ToProto() const {
  note::LayoutProto proto;
  proto.mutable_minor_to_major()->Reserve(minor_to_major().size());
  for (const auto& dim : minor_to_major_) {
    proto.add_minor_to_major(dim);
  }
  return proto;
}

std::string Layout::ToString() const {
  std::vector<std::string> dim_names;
  std::transform(minor_to_major().begin(), minor_to_major().end(),
                 std::back_inserter(dim_names),
                 [](const auto& dim) { return std::to_string(dim); });
  return paddle::string::format_string(
      "{%s}", paddle::string::join_strings(dim_names, ", ").c_str());
}

bool Layout::Valid() const { return !minor_to_major().empty(); }

bool Layout::operator==(const Layout& other) const {
  if (minor_to_major() != other.minor_to_major()) {
    return false;
  }

  return true;
}

}  // namespace piano
}  // namespace paddle
