# RayTracing

## About
3d visualizer, using raymarching technology to draw vectorized primitives

## Dependencies

* NVidia GPU and CUDA toolkit (*CUDA>=11.0.197*)
* Simple DirectMedia Layer library (*SDL>=2.0.12*)

## Building

Before building CUDA based application, you need to figure out your NVidia GPU's compute capability. 
You can determine your Compute Capability (CC) from the list:

* GTX 6xx/7xx - CC `3.0`;
* GTX 780/780ti - CC `3.5`;
* GTX 750/750ti - CC `5.0`;
* GTX 9xx - CC `5.2`;
* GTX 10xx - CC `6.1`;
* RTX 20xx - CC `7.5`;
* RTX 30xx - CC `8.6`.

If your GPU is not listed, you can evaluate your CC from *[NVidia source](https://developer.nvidia.com/cuda-gpus)*.

To let you compiler know you CC value, you need to define `CUDA_ARCH` variable with it (without delimiting point) in CMake arguments. 
For example, my GeForce GTX 1050 GPU has Compute Capability `6.1`. 
To build the application I will enter next commands in shell:

    mkdir Build
    cmake -B./Build -DCUDA_ARCH=61 -DCMAKE_BUILD_TYPE=Release .
    cmake --build ./Build --target all

## Execution

In Windows you need to add `SDL2.dll` runtime library to `./Build` directory. 
It is located in `./SDL2extra/win_x64` or `./SL2extra/win_x86`, depening on your system.

Run command `./Build/RayTracing` (in Windows - `./Build/RayTracing.exe`) with arguments:
* `--width WIDTH` - replace `WIDTH` with your desired window width;
* `--height HEIGHT` - replace `HEIGHT` with your desired window height;
* `--input FILE` - specify desired scene. Template scenes are located in `./Scenes` directory.

## Future Work

* To add more template scenes and scene creator;
* To add better illumination and materials;