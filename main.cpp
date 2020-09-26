#define _USE_MATH_DEFINES

constexpr float M_SQRT1_5f{ .447213595499957939281f };
constexpr float M_PI_2f{ 1.57079632679489661923f };

#include "rays.h"

using namespace std;

int main( int argc, char **argv ) {
    size_t Width, Height;

    string arg;
    bitset< 2 > correctInit;
    for ( int I = 1; I < argc; ++I ) {
        arg = argv[ I ];
        if ( arg.substr( 0, 2 ) == "--" ) {
            arg = arg.substr( 2, arg.length() - 2 );

            if ( arg == "width" ) {
                Width = atoi( argv[ ++I ] );
                if ( Width ) correctInit ^= 0x01;
            } else if ( arg == "height" ) {
                Height = atoi( argv[ ++I ] );
                if ( Height ) correctInit ^= 0x02;
            }

        } else if ( arg.substr( 0, 1 ) == "-" ) {
            for ( size_t n = 1; n < arg.length(); ++n )
            switch ( arg.at( n ) ) {
            }
        }
    }

    if ( !correctInit.all() ) {
        cout << "Wrong arguments!" << endl;
        return 1;
    }

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

    cudaEvent_t start, end;
    cudaEventCreate( &start );
    cudaEventCreate( &end );


    float3 LightingSourceH = float3{ 2.f * M_SQRT1_5f, 0.f, M_SQRT1_5f };

    list< primitives::base_ptr > PrimitivesH;
    {
        float3 d{ 1.f, 30.f, 1.f };
        float theta = -1.8f, w = cosf( theta / 2.f ), r = sinf( theta / 2.f ) / sqrtf( d.x * d.x + d.y * d.y + d.z * d.z );
        //PrimitivesH.push_back( primitives::rotationX::create_from( 1, cosf( theta ), sinf( theta ) ) );
        PrimitivesH.push_back( primitives::rotationQ::create_from( 1, w, r * d.x, r * d.y, r * d.z )                                );
        PrimitivesH.push_back( primitives::intersection::create_from( 1, 3 )                                                        );
        PrimitivesH.push_back( primitives::translation::create_from( 1, 0.f, 0.f, 0.f )                                             );
        PrimitivesH.push_back( primitives::cube::create_from( 50.f, 50.f, 50.f )                                                    );
        PrimitivesH.push_back( primitives::invertion::create_from( 1 )                                                              );
        PrimitivesH.push_back( primitives::unification::create_from( 1, 3 )                                                         );
        PrimitivesH.push_back( primitives::translation::create_from( 1, 0.f, 0.f, -50.f )                                           );
        PrimitivesH.push_back( primitives::sphere::create_from( 60.f )                                                              );
        PrimitivesH.push_back( primitives::translation::create_from( 1, 0.f, 0.f, 50.f )                                            );
        PrimitivesH.push_back( primitives::sphere::create_from( 40.f )                                                              );
    }

    raymarching::start_init_rays_info InfoH;
    InfoH.Width = Width;
    InfoH.Height = Height;
    InfoH.Depth = 1000;//( Width + Height ) / 2;
    InfoH.StartPos = float3{ -100.f, 0.f, 0.f };

    SDL_Point currMouse{ 0, 0 }, prevMouse;

    raymarching::Init( Width, Height, PrimitivesH.size(), SurfD );

    float cuda_time = 0.f, step = 20.f, fps = INFINITY;
    float scale = .125f, theta = 0.f, cos_theta = 1.f, sin_theta = 0.f, phi = 0.f, cos_phi = 1.f, sin_phi = 0.f;
    float cos_1 = cosf( M_PI / 180.f ), sin_1 = sinf( M_PI / 180.f ), deg = M_PI / 360.f;
    bool MIDDLE_BUTTON = false, RIGHT_BUTTON = false;

    InfoH.StartDir = float3{ scale * cos_theta * cos_phi, scale * cos_theta * sin_phi, scale * sin_theta };
    InfoH.StartWVec = float3{ scale * sin_phi, -scale * cos_phi, 0.f };
    InfoH.StartHVec = float3{ scale * sin_theta * cos_phi, scale * sin_theta * sin_phi, -scale * cos_theta };


    for ( size_t run = true, this_time = SDL_GetTicks(), prev_time = this_time;
          run;
          prev_time = this_time, this_time = SDL_GetTicks(), fps = this_time == prev_time ? 0 : 1000.f / ( this_time - prev_time ) ) {
        // Event polling
        while ( SDL_PollEvent( &Event ) ) {
            switch ( Event.type ) {
            case SDL_QUIT:
                run = false;
                break;
            case SDL_KEYDOWN:
                switch ( Event.key.keysym.sym ) {
                case SDLK_ESCAPE:
                    run = false;
                    break;
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
                        -sin_1 * LightingSourceH.x + cos_1 * LightingSourceH.y,
                        LightingSourceH.z
                    };
                    break;
                case SDLK_SPACE:
                    scale = .5f;
                    theta = 0.f;
                    cos_theta = 1.f;
                    sin_theta = 0.f;
                    phi = 0.f;
                    cos_phi = 1.f;
                    sin_phi = 0.f;

                    InfoH.StartPos = float3{ -200.f, 0.f, 0.f };
                    InfoH.StartDir = float3{ scale * cos_theta * cos_phi, scale * cos_theta * sin_phi, scale * sin_theta };
                    InfoH.StartWVec = float3{ scale * sin_phi, -scale * cos_phi, 0.f };
                    InfoH.StartHVec = float3{ scale * sin_theta * cos_phi, scale * sin_theta * sin_phi, -scale * cos_theta };
                }
                break;
            case SDL_KEYUP:
                switch ( Event.key.keysym.sym ) {
                case SDLK_ESCAPE:
                    break;
                }
                break;
            case SDL_MOUSEBUTTONDOWN:
                switch ( Event.button.button ) {
                case SDL_BUTTON_MIDDLE:
                    MIDDLE_BUTTON = true;
                    break;
                case SDL_BUTTON_RIGHT:
                    RIGHT_BUTTON = true;
                    break;
                }
                break;
            case SDL_MOUSEBUTTONUP:
                switch ( Event.button.button ) {
                case SDL_BUTTON_MIDDLE:
                    MIDDLE_BUTTON = false;
                    break;
                case SDL_BUTTON_RIGHT:
                    RIGHT_BUTTON = false;
                    break;
                }
                break;
            case SDL_MOUSEWHEEL:
                scale *= powf( 2.f, float( Event.wheel.y ) * .1f );

                InfoH.StartDir = float3{ scale * cos_theta * cos_phi, scale * cos_theta * sin_phi, scale * sin_theta };
                InfoH.StartWVec = float3{ scale * sin_phi, -scale * cos_phi, 0.f };
                InfoH.StartHVec = float3{ scale * sin_theta * cos_phi, scale * sin_theta * sin_phi, -scale * cos_theta };

                break;
            case SDL_MOUSEMOTION:
                prevMouse = currMouse;
                currMouse = SDL_Point{ Event.motion.x, Event.motion.y };

                if ( MIDDLE_BUTTON ) {
                    InfoH.StartPos.x -= ( currMouse.x - prevMouse.x ) * InfoH.StartWVec.x / scale;
                    InfoH.StartPos.y -= ( currMouse.x - prevMouse.x ) * InfoH.StartWVec.y / scale;
                    InfoH.StartPos.z -= ( currMouse.x - prevMouse.x ) * InfoH.StartWVec.z / scale;

                    InfoH.StartPos.x -= ( currMouse.y - prevMouse.y ) * InfoH.StartHVec.x / scale;
                    InfoH.StartPos.y -= ( currMouse.y - prevMouse.y ) * InfoH.StartHVec.y / scale;
                    InfoH.StartPos.z -= ( currMouse.y - prevMouse.y ) * InfoH.StartHVec.z / scale;
                } else if ( RIGHT_BUTTON ) {
                    theta += ( currMouse.y - prevMouse.y ) * deg;
                    if ( theta > M_PI_2f )
                        theta = M_PI_2f;
                    else if ( theta < -M_PI_2f )
                        theta = -M_PI_2f;
                    cos_theta = cosf( theta );
                    sin_theta = sinf( theta );

                    phi += ( currMouse.x - prevMouse.x ) * deg;
                    cos_phi = cosf( phi );
                    sin_phi = sinf( phi );
                }

                InfoH.StartDir = float3{ scale * cos_theta * cos_phi, scale * cos_theta * sin_phi, scale * sin_theta };
                InfoH.StartWVec = float3{ scale * sin_phi, -scale * cos_phi, 0.f };
                InfoH.StartHVec = float3{ scale * sin_theta * cos_phi, scale * sin_theta * sin_phi, -scale * cos_theta };

                break;
            }
        }

        LightingSourceH = float3{
            cos_1 * LightingSourceH.x - sin_1 * LightingSourceH.y,
            sin_1 * LightingSourceH.x + cos_1 * LightingSourceH.y,
            LightingSourceH.z
        };

        //InfoH.StartPos = InfoH.StartDir;
        //InfoH.StartPos.x *= -InfoH.Depth;
        //InfoH.StartPos.y *= -InfoH.Depth;
        //InfoH.StartPos.z *= -InfoH.Depth;

        // Render Scene
        cudaEventRecord( start, stream[ 0 ] );
        raymarching::Load( LightingSourceH, PrimitivesH, InfoH, stream[ 0 ] );
        raymarching::ImageProcessing( ( 256 * this_time ) / 1000, stream[ 0 ] );
        cudaEventRecord( end, stream[ 0 ] );
        cudaStreamSynchronize( stream[ 0 ] );
        cudaEventElapsedTime( &cuda_time, start, end );

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
#ifndef _DEBUG
        cout << "\rFrame per second : " << size_t( this_time - prev_time ) << "ms  \t\tTask execution : " << size_t( cuda_time ) << "ms  " << flush;
#endif
    }


    raymarching::Quit();

    cudaDestroySurfaceObject( SurfD );
    cudaGraphicsUnmapResources( 1, &res, stream[ 0 ] );
    SDL_GL_DeleteContext( GL );
    SDL_DestroyWindow( Win );

    SDL_Quit();

    return 0;
}