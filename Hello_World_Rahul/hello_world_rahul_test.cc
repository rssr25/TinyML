#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/examples/hello_world_rahul/model.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference)
{

    //set up logging
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    const tflite::Model* model = ::tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
        return 1;
    }

    // This pulls in all the operation implementations we need
    tflite::AllOpsResolver resolver;
    // Create an area of memory to use for input, output, and intermediate arrays.
    // Finding the minimum value for your model may require some trial and error.
    const int tensor_arena_size = 2 * 1024;
    uint8_t tensor_arena[tensor_arena_size];

    //build an interpreter to run the model
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, tensor_arena_size, error_reporter);
    //allocate memory from tensor arena for model's tensors
    interpreter.AllocateTensors();

    //pointer to model's input tensor
    TfLiteTensor* input = interpreter.input(0);

    // Make sure the input has the properties we expect
    TF_LITE_MICRO_EXPECT_NE(nullptr, input);
    // The property "dims" tells us the tensor's shape. It has one element for
    // each dimension. Our input is a 2D tensor containing 1 element, so "dims"
    // should have size 2.
    TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
    // The value of each element gives the length of the corresponding tensor.
    // We should expect two single element tensors (one is contained within the
    // other).
    TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
    TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
    // The input is a 32 bit floating point value
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);


    //Running the inference
    // Provide an input value
    input->data.f[0] = 1.;
    // Run the model on this input and check that it succeeds
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed\n");
    }
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

    //check the output
    TfLiteTensor* output = interpreter.output(0);
    TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
    TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
    TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);

    // Obtain the output value from the tensor
    float value = output->data.f[0];
    // Check that the output value is within 0.05 of the expected value
    TF_LITE_MICRO_EXPECT_NEAR(0., value, 0.05);

    // Run inference on several more values and confirm the expected outputs
    input->data.f[0] = 1.;
    interpreter.Invoke();
    value = output->data.f[0];
    TF_LITE_MICRO_EXPECT_NEAR(0.841, value, 0.05);
    input->data.f[0] = 3.;
    interpreter.Invoke();
    value = output->data.f[0];
    TF_LITE_MICRO_EXPECT_NEAR(0.141, value, 0.05);
    input->data.f[0] = 5.;
    interpreter.Invoke();
    value = output->data.f[0];
    TF_LITE_MICRO_EXPECT_NEAR(-0.959, value, 0.05);
}

TF_LITE_MICRO_TESTS_END
