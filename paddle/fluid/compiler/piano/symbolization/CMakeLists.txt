cc_library(note_builder SRCS note_builder.cc DEPS string_helper note_opcode piano_data_description)
cc_test(note_builder_test SRCS note_builder_test.cc DEPS note_builder)

cc_library(piano_meta_op SRCS meta_op.cc shape_inference.cc DEPS note_opcode note_builder note_template_util)
cc_test(piano_meta_op_test SRCS meta_op_test.cc DEPS piano_meta_op)
cc_test(symbolization_shape_inference_test SRCS shape_inference_test.cc DEPS piano_meta_op)
