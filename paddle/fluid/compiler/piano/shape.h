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
#include "paddle/fluid/compiler/piano/layout.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace piano {

// A Shape describes the rank, size, and data type of an N-dimensional
// array.
// For tuples, it describes the structure (number of elements and nesting).
class Shape {
 public:
  Shape() = default;

  // Construct a shape from a ShapeProto.
  explicit Shape(const note::ShapeProto& proto);

  // Construct a shape with detail data member
  Shape(note::ElementTypeProto element_type,
        const std::vector<int64_t>& dimensions,
        std::vector<Shape> tuple_shapes = {})
      : element_type_(element_type),
        dimensions_(dimensions.begin(), dimensions.end()),
        tuple_shapes_(std::move(tuple_shapes)) {}

  // Pack data member into a note::ShapeProto
  note::ShapeProto ToProto() const;

  // Returns a human-readable string that represents the given shape.
  // e.g. "f32[2, 3]{0, 1}" denotes an float32 array
  // with dimensions size [2, 3], and its layout is {0, 1}
  std::string ToString() const;

  // Returns the rank of the given shape. Shape must be an array.
  int64_t Rank() const {
    PADDLE_ENFORCE_NE(
        element_type(), note::ELEMENT_TYPE_TUPLE,
        platform::errors::InvalidArgument(
            "Non-array do not have a rank, shape: %s", ToString()));
    return dimensions_.size();
  }

  // Returns whether the shape is of the specified type
  bool IsArray() const { return !IsTuple(); }
  bool IsTuple() const { return element_type() == note::ELEMENT_TYPE_TUPLE; }

  // an option indicates how to compare two shapes
  struct CompareOption {
    CompareOption& IgnoreElementType() {
      ignore_element_type = true;
      return *this;
    }

    CompareOption& IgnoreDimensions() {
      ignore_dimensions = true;
      return *this;
    }

    CompareOption& IgnoreLayout() {
      ignore_layout = true;
      return *this;
    }

    bool ignore_element_type = false;
    bool ignore_dimensions = false;
    bool ignore_layout = false;
  };

  // overload euqal operator that tests all fields of the shape are the same
  bool EqualTo(const Shape& other, const CompareOption& option) const;
  bool operator==(const Shape& other) const {
    static CompareOption default_option;
    return EqualTo(other, default_option);
  }
  bool operator!=(const Shape& other) const { return !(*this == other); }

  // The following methods for accessing the data member of a Shape object
  // stores.
  //
  // Methods for accessing the element type.
  note::ElementTypeProto element_type() const { return element_type_; }
  void set_element_type(note::ElementTypeProto value) { element_type_ = value; }

  // Methods for accessing the dimensions array.
  std::vector<int64_t>* mutable_dimensions() { return &dimensions_; }
  const std::vector<int64_t>& dimensions() const { return dimensions_; }

  // Methods for accessing the tuple subshapes. This field only non-empty for
  // tuple shapes.
  std::vector<Shape>* mutable_tuple_shapes() { return &tuple_shapes_; }
  const std::vector<Shape>& tuple_shapes() const { return tuple_shapes_; }

  // Methods for accessing the layout field.
  Layout* mutable_layout() { return &layout_; }
  const Layout& layout() const { return layout_; }
  bool has_layout() const { return layout_.Valid(); }

 private:
  // The element type of this shape (tuple, array, etc).
  note::ElementTypeProto element_type_ = note::INVALID_ELEMENT_TYPE;

  // The array bounds of the dimensions.
  std::vector<int64_t> dimensions_;

  // The tuple element subshapes. This is non-empty only for tuple shapes.
  std::vector<Shape> tuple_shapes_;

  // The layout of the shape. Only relevant for arrays.
  Layout layout_;
};

// Shape of the parameters and output of a note funtion, as its signature.
class Signature {
 public:
  Signature() = default;

  // Creates a Signature from a SignatureProto protobuf.
  explicit Signature(const note::SignatureProto& proto);

  Signature(std::vector<Shape>&& parameters,
            std::vector<std::string>&& parameter_names, const Shape& result)
      : parameters_(parameters),
        parameter_names_(parameter_names),
        result_(result) {}

  note::SignatureProto ToProto() const;

  std::string ToString() const;

  // The following methods for accessing the data member of a Signature object
  // stores.
  //
  // Methods for accessing and manipulating the Shape of the parameters.
  std::vector<Shape>* mutable_parameters() { return &parameters_; }
  const std::vector<Shape>& parameters() const { return parameters_; }

  // Methods for accessing and manipulating the Shape of the result.
  Shape* mutable_result() { return &result_; }
  const Shape& result() const { return result_; }

  // Methods for accessing and manipulating the names of the parameters.
  std::vector<std::string>* mutable_parameter_names() {
    return &parameter_names_;
  }
  const std::vector<std::string>& parameter_names() const {
    return parameter_names_;
  }

 private:
  // The shapes of the parameters of the function represented by this object.
  std::vector<Shape> parameters_;

  // The names of the parameters of the function represented by this object.
  std::vector<std::string> parameter_names_;

  // The shape of the result of the function represented by this object.
  Shape result_;
};

}  // namespace piano
}  // namespace paddle
