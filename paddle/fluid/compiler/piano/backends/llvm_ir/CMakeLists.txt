cc_library(llvm_utils SRCS llvm_utils.cc DEPS llvm)

cc_library(primitive_ir_emitter SRCS primitive_ir_emitter.cc DEPS llvm)
cc_library(gpu_primitive_ir_emitter SRCS gpu_primitive_ir_emitter.cc DEPS llvm)
cc_library(nvptx_primitive_ir_emitter SRCS nvptx_primitive_ir_emitter.cc
    DEPS gpu_primitive_ir_emitter primitive_ir_emitter llvm_utils llvm)

cc_test(nvptx_primitive_ir_emitter_test SRCS nvptx_primitive_ir_emitter_test.cc
        DEPS nvptx_primitive_ir_emitter)
target_link_libraries(nvptx_primitive_ir_emitter_test ${LLVM_LIBS})

cc_library(ir_emitter SRCS ir_emitter.cc DEPS llvm)
cc_library(gpu_ir_emitter SRCS gpu_ir_emitter.cc DEPS ir_emitter llvm_utils llvm)
cc_library(nvptx_ir_emitter SRCS nvptx_ir_emitter.cc DEPS gpu_ir_emitter nvptx_primitive_ir_emitter llvm)

cc_test(nvptx_ir_emitter_test SRCS nvptx_ir_emitter_test.cc
        DEPS nvptx_ir_emitter note_ir note_proto)
target_link_libraries(nvptx_ir_emitter_test ${LLVM_LIBS})

cc_library(llvm_compiler SRCS llvm_compiler.cc DEPS llvm)
cc_library(nvtpx_compiler SRCS nvptx_compiler.cc DEPS llvm)

cc_test(nvptx_compiler_test SRCS nvptx_compiler_test.cc
        DEPS nvtpx_compiler llvm_compiler)
target_link_libraries(nvptx_compiler_test ${LLVM_LIBS})
