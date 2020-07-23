#include "tensorflow/lite/micro/examples/hello_world_rahul/main_functions.h"


int main(int argc, char* argv[])
{
    setup();
    while(true)
    {
        loop();
    }
}