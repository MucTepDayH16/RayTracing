#define _USE_MATH_DEFINES

#include <iostream>
#include <list>
#include <cmath>
#include <memory>

#include <SDL.h>
#include <SDL_opengl.h>

#include "rays.h"

int main( int argc, char **argv ) {
    size_t CudaStreamNum = 10;
    cudaStream_t *stream = new cudaStream_t[ CudaStreamNum ];
    for ( size_t i = 0; i < CudaStreamNum; ++i )
        cudaStreamCreate( stream + i );

    float3 LightingSourceH = float3{ M_SQRT1_2, M_SQRT1_2, 0.f };

    cuda::pointer< float3 > LightingSourceD;
    LightingSourceD.CopyFrom( &LightingSourceH );

    std::cout << sizeof primitives::base << std::endl;
    std::cout << sizeof primitives::sphere << std::endl;

    std::list< primitives::base_ptr > PrimitivesH;
    {
        primitives::base *ptr = new primitives::sphere( float3{ 0.f, 0.f, 0.f }, 1.f );
        PrimitivesH.push_back( ptr );
    }
    size_t PrimitivesNum = PrimitivesH.size(), i = 0;
    cuda::pointer< primitives::base > PrimitivesD( PrimitivesNum );

    for ( primitives::base_ptr ptr : PrimitivesH ) {
        PrimitivesD.CopyFrom( ptr, i );
    }

    size_t Width = 800, Height = 600;

    SDL_Init( SDL_INIT_EVERYTHING );
    SDL_Window *Win = SDL_CreateWindow( "GL", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, Width, Height, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL );
    SDL_GLContext GL = SDL_GL_CreateContext( Win );
    SDL_Event Event;

    raymarching::start( LightingSourceD, PrimitivesD, Width, Height );

    for ( size_t run = true, this_time = SDL_GetTicks(), prev_time = this_time, fps = 0;
          run;
          prev_time = this_time, this_time = SDL_GetTicks(), fps = this_time == prev_time ? MAXDWORD64 : 1000 / ( this_time - prev_time ) ) {
        // Event polling
        while ( SDL_PollEvent( &Event ) ) {
            switch ( Event.type ) {
            case SDL_QUIT:
                run = false;
                break;
            }
        }

        // Scene draw
        glClearColor( .5f, .5f, 0.f, 1.f );
        glClear( GL_COLOR_BUFFER_BIT );

        // Scene update
        SDL_GL_SwapWindow( Win );
    }

    SDL_GL_DeleteContext( GL );
    SDL_DestroyWindow( Win );

    return 0;
}