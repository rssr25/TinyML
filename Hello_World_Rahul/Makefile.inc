HELLO_WORLD_RAHUL_TEST_SRCS := \
tensorflow/lite/micro/examples/hello_world_rahul/hello_world_rahul_test.cc \
tensorflow/lite/micro/examples/hello_world_rahul/model.cc

HELLO_WORLD_RAHUL_TEST_HDRS := \
tensorflow/lite/micro/examples/hello_world_rahul/model.h

OUTPUT_HANDLER_TEST_SRCS := \
tensorflow/lite/micro/examples/hello_world_rahul/output_handler_test.cc \
tensorflow/lite/micro/examples/hello_world_rahul/output_handler.cc

OUTPUT_HANDLER_TEST_HDRS := \
tensorflow/lite/micro/examples/hello_world_rahul/output_handler.h \
tensorflow/lite/micro/examples/hello_world_rahul/constants.h


HELLO_WORLD_RAHUL_SRCS := \
tensorflow/lite/micro/examples/hello_world_rahul/main.cc \
tensorflow/lite/micro/examples/hello_world_rahul/main_functions.cc \
tensorflow/lite/micro/examples/hello_world_rahul/model.cc \
tensorflow/lite/micro/examples/hello_world_rahul/output_handler.cc \
tensorflow/lite/micro/examples/hello_world_rahul/constants.cc


HELLO_WORLD_RAHUL_HDRS := \
tensorflow/lite/micro/examples/hello_world_rahul/model.h \
tensorflow/lite/micro/examples/hello_world_rahul/output_handler.h \
tensorflow/lite/micro/examples/hello_world_rahul/constants.h \
tensorflow/lite/micro/examples/hello_world_rahul/main_functions.h


#Find any platform - specific rules for this example.
include $(wildcard tensorflow/lite/micro/examples/hello_world_rahul/*/Makefile.inc)

# Tests loading and running the sine model.
$(eval $(call microlite_test,hello_world_rahul_test,\
$(HELLO_WORLD_RAHUL_TEST_SRCS),$(HELLO_WORLD_RAHUL_TEST_HDRS)))

# Tests producing an output.
$(eval $(call microlite_test,output_handler_test,\
$(OUTPUT_HANDLER_TEST_SRCS),$(OUTPUT_HANDLER_TEST_HDRS)))


# Builds a standalone binary.
$(eval $(call microlite_test,hello_world_rahul,\
$(HELLO_WORLD_RAHUL_SRCS),$(HELLO_WORLD_RAHUL_HDRS)))
