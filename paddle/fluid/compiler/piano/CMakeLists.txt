add_subdirectory(backends)
add_subdirectory(note)
add_subdirectory(symbolization)

cc_library(piano_data_description SRCS layout.cc shape.cc DEPS string_helper note_proto)
cc_test(piano_layout_test SRCS layout_test.cc DEPS piano_data_description)
cc_test(piano_shape_test SRCS shape_test.cc DEPS piano_data_description)
