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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/note_builder.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace piano {

class PianoOpKernelContext;
class PianoOpMaker;

class PianoOpRegistry final {
 public:
  using OpKernelFunc = std::function<void(const PianoOpKernelContext&)>;
  using OpKernelMap = std::unordered_map<std::string, OpKernelFunc>;

  // Register a Piano backend.
  // `name` is the backend name. `supported_types` is data type list,
  // this backend can only accept the data type in list. `filter_func` is
  // a function, return false if the backend refuse this op.
  using BackendFilterFunc = bool (*)(Operand*);
  static void RegisterBackend(
      const std::string& backend_name,
      const std::unordered_set<note::ElementTypeProto>& supported_types,
      BackendFilterFunc filter_func);

  static inline bool IsBackend(const std::string& backend_name) {
    return Instance().backend_.count(backend_name) > 0;
  }

  static std::vector<std::string> AllBackendNames();

  static const std::unordered_set<note::ElementTypeProto>& BackendDataTypes(
      const std::string& backend_name);

  // Piano Op interface
  static inline bool IsPianoOp(const std::string& op_type) {
    return Instance().ops_.count(op_type) > 0;
  }

  static std::vector<std::string> AllPianoOps();

  static bool HasAllowBackendList(const std::string& op_type);

  static std::vector<std::string> AllowBackendList(const std::string& op_type) {
    return HasAllowBackendList(op_type)
               ? Instance().ops_.at(op_type)->allow_backend_list
               : AllBackendNames();
  }

  static const std::unordered_set<note::ElementTypeProto>& PianoOpDataTypes(
      const std::string& op_type);

  static const framework::AttributeMap& Attrs(const std::string& op_type);

  static void RegisterKernel(const std::string& op_type,
                             const std::string& library_type,
                             OpKernelFunc func) {
    // save kernel information into kernel_ map
    Instance().ops_.at(op_type)->kernel_.emplace(library_type, func);
  }

  static const OpKernelMap& AllPianoOpKernels(const std::string& op_type);

 private:
  // Declare PianoOpMaker friend class so that AddAttr can add attribute into
  // ops_'s attrs value.
  // Why not define an AddAttr function in PianoOpRegistry? Only PianoOpMaker
  // can access attribute.
  friend class PianoOpMaker;

  // register class
  template <typename OpMakeType, typename KernelType>
  friend class PianoOpRegistrar;

  static PianoOpRegistry& Instance() {
    static PianoOpRegistry r;
    return r;
  }

  PianoOpRegistry() = default;
  ~PianoOpRegistry() = default;

  DISABLE_COPY_AND_ASSIGN(PianoOpRegistry);

  // Describes a Piano backend
  struct Backend {
    std::string name;
    std::unordered_set<note::ElementTypeProto> supported_types;

    // A filter function used to exclude or modify operator
    // registrations on the device. If nullptr, the backend
    // accept all op, else it should return false if the op
    // cannot register at this backend.
    // The function may modify operator to adapt the backend.
    BackendFilterFunc filter_func = nullptr;
  };

  // Map from backend name to its descriptor
  std::unordered_map<std::string, std::unique_ptr<Backend>> backend_;

  // Describes a Paddle operator that can be compiled to Piano
  struct OpRegistration {
    std::string op_type;
    std::unordered_set<note::ElementTypeProto> supported_types;

    bool has_allow_backend_list = false;
    std::vector<std::string> allow_backend_list;

    // Different to OpProto::attrs, these attribute are only used for
    // Piano, which can be obtained at Piano compile time.
    framework::AttributeMap attrs;

    std::unique_ptr<PianoOpMaker> maker;

    // Piano Op kernel map, the key is library name and its value is a
    // kernel function, the kernel function override the "Compile"
    // interface of "PianoOpKernel" class.
    OpKernelMap kernel_;
  };

  // Map from operator name to its descriptor
  std::unordered_map<std::string, std::unique_ptr<OpRegistration>> ops_;
};

// just used for mark final keyword
class PianoOpMakerBase {
 public:
  virtual void BindOp(const std::string& op_type) = 0;
  virtual ~PianoOpMakerBase() = default;
};

class PianoOpMaker : public PianoOpMakerBase {
 public:
  virtual void Make() = 0;

  virtual ~PianoOpMaker() = default;

  // Do not rewrite this API in derived class!
  void BindOp(const std::string& op_type) final {
    this->op_ = PianoOpRegistry::Instance().ops_.at(op_type).get();
  }

 protected:
  // cover the old one if existed a same name attribute
  // Do not rewrite this API in derived class!
  template <typename T>
  void AddAttr(const std::string& name, const T& val) {
    op_->attrs.emplace(name, val);
  }

  void SetAllowBackendList(const std::vector<std::string>& backends) {
    op_->has_allow_backend_list = true;
    op_->allow_backend_list = backends;
  }

  void SetDataTypes(
      const std::unordered_set<note::ElementTypeProto>& data_types) {
    op_->supported_types.insert(data_types.cbegin(), data_types.cend());
  }

 private:
  PianoOpRegistry::OpRegistration* op_;
};

template <class OpMakeType, class KernelType>
class PianoOpRegistrar final : public framework::Registrar {
 public:
  PianoOpRegistrar(const char* op_type, const char* library_type) {
    using paddle::framework::OpInfoMap;
    PADDLE_ENFORCE_EQ(OpInfoMap::Instance().Has(op_type), true,
                      platform::errors::NotFound(
                          "Piano OP should registered in Paddle before, "
                          "but %s not. Please use \"REGISTER_OPERATOR\" "
                          "before register Piano OP.",
                          op_type));

    PADDLE_ENFORCE_EQ(PianoOpRegistry::IsPianoOp(op_type), false,
                      platform::errors::AlreadyExists(
                          "Piano OP %s has been registered.", op_type));

    // bind PianoOpMaker class for add Piano attribute later
    // Do need check whether OpMakeType derive from PianoOpMaker?
    static_assert(std::is_base_of<PianoOpMaker, OpMakeType>::value,
                  "The OpMaker class is not derived from PianoOpMaker.");

    // create and bind OpRegistration
    auto& registry = PianoOpRegistry::Instance();
    registry.ops_.emplace(op_type, new PianoOpRegistry::OpRegistration);
    auto& op_reg = registry.ops_.at(op_type);
    op_reg->op_type = op_type;

    // bind PianoOpMaker class for add Piano attribute later
    op_reg->maker.reset(new OpMakeType);
    op_reg->maker->BindOp(op_type);
    // TODO(jiangcheng05): here invoke Make() is not a good idea
    op_reg->maker->Make();

    PianoOpRegistry::RegisterKernel(
        op_type, library_type,
        [](const PianoOpKernelContext& ctx) { KernelType().Compile(ctx); });
  }
};

#define REGISTER_PIANO_OP(op_type, op_maker, op_kernel) \
  REGISTER_PIANO_OP_EX(op_type, PLAIN, op_maker, op_kernel)

#define REGISTER_PIANO_OP_EX(TYPE, LIB, MAKER, KERNEL)         \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                              \
      __reg_piano_op__##TYPE##_##LIB,                          \
      "REGISTER_PIANO_OP must be called in global namespace"); \
  static ::paddle::piano::PianoOpRegistrar<MAKER, KERNEL>      \
      __piano_op_registrar__##TYPE##_##LIB##__(#TYPE, #LIB);   \
  int TouchPianoOpRegistrar_##TYPE##_##LIB() {                 \
    __piano_op_registrar__##TYPE##_##LIB##__.Touch();          \
    return 0;                                                  \
  }

class BackendRegistrar final : public framework::Registrar {
 public:
  BackendRegistrar(
      const char* backend_name,
      const std::unordered_set<note::ElementTypeProto>& supported_types,
      PianoOpRegistry::BackendFilterFunc filter_func = nullptr) {
    PianoOpRegistry::RegisterBackend(backend_name, supported_types,
                                     filter_func);
  }
};

#define REGISTER_PIANO_BACKEND(NAME, ...)                           \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                   \
      __reg_piano_backend__##NAME,                                  \
      "REGISTER_PIANO_BACKEND must be called in global namespace"); \
  static ::paddle::piano::BackendRegistrar                          \
      __piano_backend_registrar__##NAME##__(#NAME, __VA_ARGS__);    \
  int TouchBackendRegistrar_##NAME() {                              \
    __piano_backend_registrar__##NAME##__.Touch();                  \
    return 0;                                                       \
  }

}  // namespace piano
}  // namespace paddle
