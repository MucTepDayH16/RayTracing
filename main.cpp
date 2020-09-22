#define _USE_MATH_DEFINES

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

    size_t Width = 600, Height = 600;

    SDL_Init( SDL_INIT_EVERYTHING );
    SDL_Window *Win = SDL_CreateWindow( "GL", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, Width, Height, SDL_WINDOW_SHOWN | SDL_WINDOW_BORDERLESS | SDL_WINDOW_OPENGL );
    SDL_GLContext GL = SDL_GL_CreateContext( Win );
    glClearColor( 0.0, 0.0, 0.0, 0.0 );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( 0.0, Width, Height, 1.0, -1.0, 1.0 );
    glEnable( GL_BLEND );
    glEnable( GL_TEXTURE_2D );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );


    //SDL_Surface *surf = IMG_Load( "image.png" ); //SDL_CreateRGBSurface( 0, Width, Height, 32, 0xff000000, 0x00ff0000, 0x0000ff00, 0x000000ff );
    //GLuint *Pixels = ( GLuint* ) surf->pixels;
    //float x = .5f;

    ////SDL_LockSurface( surf );
    ////size_t temp;
    ////for ( size_t y = 0; y < Height; ++y ) {
    ////    for ( size_t x = 0; x < Width; ++x ) {
    ////        temp = ( x + y );
    ////        Pixels[ y * Width + x ] = 0x000000ff | ( ( uint32_t( temp & 0xff ) * 0x10101 ) << 8 );
    ////    }
    ////}
    ////SDL_UnlockSurface( surf );

    ////IMG_SavePNG( surf, "image.png" );

    GLuint tex;
    glGenTextures( 1, &tex );
    glBindTexture( GL_TEXTURE_2D, tex );
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, Width, Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr );

    cudaGraphicsResource *res;
    cudaGraphicsGLRegisterImage( &res, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore );
    cudaGraphicsMapResources( 1, &res, stream[ 0 ] );

    cudaArray_t pixels_d;
    cudaGraphicsSubResourceGetMappedArray( &pixels_d, res, 0, 0 );

    cudaResourceDesc desc;
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = pixels_d;

    cudaSurfaceObject_t surf_d;
    cudaCreateSurfaceObject( &surf_d, &desc );

    SDL_Event Event;

    raymarching::Start( LightingSourceD, PrimitivesD, Width, Height );

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

        // Render Scene
        raymarching::ImageProcessing( surf_d, Width, Height, ( 256 * this_time ) / 1000, stream[ 0 ] );

        // Draw Scene
        glClear( GL_COLOR_BUFFER_BIT );
        glBindTexture( GL_TEXTURE_2D, tex );
        glBegin( GL_QUADS );
            glTexCoord2i( 0, 0 );
            glVertex2i( 0, 0 );
            glTexCoord2i( 1, 0 );
            glVertex2i( Width, 0 );
            glTexCoord2i( 1, 1 );
            glVertex2i( Width, Height );
            glTexCoord2i( 0, 1 );
            glVertex2i( 0, Height );
        glEnd();

        // Scene update
        SDL_GL_SwapWindow( Win );
        std::cout << fps << std::endl;
    }

    //SDL_FreeSurface( surf );
    cudaDestroySurfaceObject( surf_d );
    cudaGraphicsUnmapResources( 1, &res, stream[ 0 ] );
    SDL_GL_DeleteContext( GL );
    SDL_DestroyWindow( Win );

    SDL_Quit();

    return 0;
}