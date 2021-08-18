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

// A Layout describes how an array is represented in memory
class Layout {
 public:
  Layout() = default;

  // Construct a layout from a LayoutProto.
  explicit Layout(const note::LayoutProto& proto);

  // Constructs a layout with the given minor-to-major order.
  explicit Layout(const std::vector<int64_t>& minor_to_major)
      : minor_to_major_(minor_to_major.begin(), minor_to_major.end()) {}

  // Returns a LayoutProto representation of the Layout.
  note::LayoutProto ToProto() const;

  // Returns a human-readable string that represents this layout.
  std::string ToString() const;

  // Return whether this layout is valid
  bool Valid() const;

  // Return whether this layout euqal to the other
  bool operator==(const Layout& other) const;
  bool operator!=(const Layout& other) const { return !(*this == other); }

  // The following methods for accessing the data member of a Layout object
  // stores.
  //
  // Methods for accessing the minor-to-major array.
  std::vector<int64_t>* mutable_minor_to_major() { return &minor_to_major_; }
  const std::vector<int64_t>& minor_to_major() const { return minor_to_major_; }

  void Clear() { mutable_minor_to_major()->clear(); }

 private:
  // A map from physical dimension numbers to logical dimension numbers.
  // The first element is the most minor physical dimension (fastest varying
  // index) and the last the most major (slowest varying index). The contents of
  // the vector are the indices of the *logical* dimensions in the shape.
  //
  // For example, in shape f32[8,100,100,3]{3,0,2,1}, the logical dimensions
  // are [8,100,100,3] and minor_to_major_ is {3,0,2,1}.
  // So, the most minor physical dimension is [8,100,100,3][3], which is size 3.
  // The second most minor is [8,100,100,3][0], which is size 8.
  // The third most minor is [8,100,100,3][2], which is size 100.
  // And the major dim is [8,100,100,3][1], which is size 100.
  std::vector<int64_t> minor_to_major_;
};

}  // namespace piano
}  // namespace paddle
