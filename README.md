# RayTracing

## About
3d visualizer, using raymarching technology to draw vectorized primitives

## Dependencies

* NVidia GPU and CUDA toolkit (*CUDA>=11.0.197*)
* Simple DirectMedia Layer library (*SDL>=2.0.12*)

## Building

To build the application I will enter next commands in shell:

    mkdir Build
    cmake -B./Build -DCMAKE_BUILD_TYPE=Release .
    cmake --build ./Build --target all

## Execution

On Windows you need to add `SDL2.dll` runtime library to `./Build` directory.
It is located in `./SDL2extra/win_x64` or `./SL2extra/win_x86`, depending on your system.

Run `./Build/RayTracing` (`./Build/RayTracing.exe` on Windows) with arguments:
* `--width WIDTH` - replace `WIDTH` with your desired window width;
* `--height HEIGHT` - replace `HEIGHT` with your desired window height;
* ~`--input FILE` - specify desired scene. Template scenes are located in `./Scenes` directory.~

## Future Work

* To test Linux-compliance;
* To add OpenCL compute method for Radeon GPU's compatibility; 
* To add more template scenes and scene creator;
* To add better illumination and materials.
