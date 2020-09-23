#define _USE_MATH_DEFINES

#include "rays.h"

int main( int argc, char **argv ) {
    size_t Width = 800, Height = 600;

    size_t CudaStreamNum = 10;
    cudaStream_t *stream = new cudaStream_t[ CudaStreamNum ];
    for ( size_t i = 0; i < CudaStreamNum; ++i )
        cudaStreamCreate( stream + i );

    SDL_Init( SDL_INIT_EVERYTHING );
    SDL_Window *Win = SDL_CreateWindow( "GL", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, Width, Height, SDL_WINDOW_SHOWN | SDL_WINDOW_BORDERLESS | SDL_WINDOW_OPENGL );
    SDL_GLContext GL = SDL_GL_CreateContext( Win );
    glClearColor( 0.f, 0.f, 0.f, 1.f );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( 0.f, Width, Height, 1.f, -1.f, 1.f );
    glEnable( GL_BLEND );
    glEnable( GL_TEXTURE_2D );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

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

    cudaSurfaceObject_t SurfD;
    cudaCreateSurfaceObject( &SurfD, &desc );

    SDL_Event Event;


    float3 LightingSourceH = float3{ M_SQRT1_2, 0.f, M_SQRT1_2 };

    std::list< primitives::base_ptr > PrimitivesH;
    {
        PrimitivesH.push_back( primitives::cube::create_from( float3{ 500.f, 0.f, 0.f }, float3{ 50.f, 50.f, 50.f } ) );
        PrimitivesH.push_back( primitives::sphere::create_from( float3{ 500.f, 100.f, 0.f }, 40.f ) );
        PrimitivesH.push_back( primitives::sphere::create_from( float3{ 500.f, 200.f, 0.f }, 30.f ) );
        PrimitivesH.push_back( primitives::sphere::create_from( float3{ 500.f, 300.f, 0.f }, 20.f ) );
        PrimitivesH.push_back( primitives::sphere::create_from( float3{ 500.f, 400.f, 0.f }, 10.f ) );
        PrimitivesH.push_back( primitives::sphere::create_from( float3{ 500.f, -100.f, 0.f }, 40.f ) );
        PrimitivesH.push_back( primitives::sphere::create_from( float3{ 500.f, -200.f, 0.f }, 30.f ) );
        PrimitivesH.push_back( primitives::sphere::create_from( float3{ 500.f, -300.f, 0.f }, 20.f ) );
        PrimitivesH.push_back( primitives::sphere::create_from( float3{ 500.f, -400.f, 0.f }, 10.f ) );
    }

    raymarching::start_init_rays_info InfoH;
    InfoH.Width = Width;
    InfoH.Height = Height;
    InfoH.Depth = 100;//( Width + Height ) / 2;
    InfoH.StartPos = float3{ 0.f, 0.f, 0.f };
    InfoH.StartDir = float3{ 1.f, 0.f, 0.f };
    InfoH.StartWVec = float3{ 0.f, -1.f, 0.f };
    InfoH.StartHVec = float3{ 0.f, 0.f, -1.f };

    SDL_Point currMouse, prevMouse;

    raymarching::Init( Width, Height, PrimitivesH.size(), SurfD );
    std::cout << std::endl;

    float scale = 1.f, theta, cos_theta = 1.f, sin_theta = 0.f, phi, cos_phi = 1.f, sin_phi = 0.f;
    float cos_1 = cosf( M_PI / 180.f ), sin_1 = sinf( M_PI / 180.f );


    for ( size_t run = true, this_time = SDL_GetTicks(), prev_time = this_time, fps = 0;
          run;
          prev_time = this_time, this_time = SDL_GetTicks(), fps = this_time == prev_time ? MAXDWORD64 : 1000 / ( this_time - prev_time ) ) {
        // Event polling
        while ( SDL_PollEvent( &Event ) ) {
            switch ( Event.type ) {
            case SDL_QUIT:
                run = false;
                break;
            case SDL_KEYDOWN:
                switch ( Event.key.keysym.sym ) {
                case SDLK_LEFT:
                    LightingSourceH = float3{
                        cos_1 * LightingSourceH.x - sin_1 * LightingSourceH.y,
                        sin_1 * LightingSourceH.x + cos_1 * LightingSourceH.y,
                        LightingSourceH.z
                    };
                    break;
                case SDLK_RIGHT:
                    LightingSourceH = float3{
                        cos_1 * LightingSourceH.x + sin_1 * LightingSourceH.y,
                        - sin_1 * LightingSourceH.x + cos_1 * LightingSourceH.y,
                        LightingSourceH.z
                    };
                    break;
                }
                break;
            case SDL_MOUSEWHEEL:
                scale *= powf( 2.f, float( Event.wheel.y ) * .1f );
                break;
            case SDL_MOUSEMOTION:
                currMouse = SDL_Point{ Event.motion.x, Event.motion.y };

                theta = M_PI * ( .5f - float( currMouse.y ) / Height );
                cos_theta = cosf( theta );
                sin_theta = sinf( theta );

                phi = M_PI * ( .5f - float( currMouse.x ) / Width );
                cos_phi = cosf( phi );
                sin_phi = sinf( phi );

                break;
            }
        }

        InfoH.StartDir = float3{ scale * cos_theta * cos_phi, scale * cos_theta * sin_phi, scale * sin_theta };
        InfoH.StartWVec = float3{ scale * sin_phi, -scale * cos_phi, 0.f };
        InfoH.StartHVec = float3{ scale * sin_theta * cos_phi, scale * sin_theta * sin_phi, -scale * cos_theta };

        // Render Scene
        raymarching::Load( LightingSourceH, PrimitivesH, InfoH, stream[ 0 ] );
        raymarching::ImageProcessing( ( 256 * this_time ) / 1000, stream[ 0 ] );

        // Draw Scene
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

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


    raymarching::Quit();

    cudaDestroySurfaceObject( SurfD );
    cudaGraphicsUnmapResources( 1, &res, stream[ 0 ] );
    SDL_GL_DeleteContext( GL );
    SDL_DestroyWindow( Win );

    SDL_Quit();

    return 0;
}