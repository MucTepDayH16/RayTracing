#define _USE_MATH_DEFINES

constexpr float M_SQRT1_5f{ .447213595499957939281f };
constexpr float M_PI_2f{ 1.57079632679489661923f };

#include <cuda_rays.cuh>

using namespace std;

int main( int argc, char **argv ) {
    size_t Width, Height;
    string File;

    string arg;
    bitset< 3 > correctInit;
    for ( int I = 1; I < argc; ++I ) {
        arg = argv[ I ];
        if ( arg.substr( 0, 2 ) == "--" ) {
            arg = arg.substr( 2, arg.length() - 2 );

            if ( arg == "width" ) {
                Width = atoi( argv[ ++I ] );
                if ( Width ) correctInit ^= 1 << 0;
            } else if ( arg == "height" ) {
                Height = atoi( argv[ ++I ] );
                if ( Height ) correctInit ^= 1 << 1;
            } else if ( arg == "input" ) {
                File = argv[ ++I ];
                if ( File.size() ) correctInit ^= 1 << 2;
            }

        } else if ( arg.substr( 0, 1 ) == "-" ) {
            for ( size_t n = 1; n < arg.length(); ++n )
                switch ( arg.at( n ) ) {
                }
        }
    }
    

    SDL_Init( SDL_INIT_EVERYTHING );
    SDL_DisplayMode DM;
    SDL_GetCurrentDisplayMode( 0, &DM );
    if ( !correctInit[ 0 ] ) Width = DM.w;
    if ( !correctInit[ 1 ] ) Height = DM.h;

    SDL_Window *Win = SDL_CreateWindow(
        "GL",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        Width, Height,
        SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_BORDERLESS );
    SDL_GLContext GL = SDL_GL_CreateContext( Win );
    glClearColor( 0.f, 0.f, 0.f, 1.f );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( 0.f, Width, Height, 1.f, -1.f, 1.f );
    glEnable( GL_BLEND );
    glEnable( GL_TEXTURE_2D );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

    SDL_Event Event;
    SDL_Point currMouse{ 0, 0 }, prevMouse;
    
    null::raymarching* CUDA = new cuda::raymarching;
    CUDA->Init( Width, Height );
    
    GLuint tex;
    glGenTextures( 1, &tex );
    glBindTexture( GL_TEXTURE_2D, tex );
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, Width, Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr );

    CUDA->SetTexture( tex );

    /*if ( correctInit[ 2 ] )
        raymarching::PrimitivesI( PrimitivesH, File );
    else {
        // INFINITY
        //PrimitivesH.push_back(
        //    primitives::komplemento::create_from( 1 )
        //);
        //PrimitivesH.push_back(
        //    primitives::senfina_ripeto::create_from( 1, 0.f, 0.f, 100.f )
        //);
        //PrimitivesH.push_back(
        //    primitives::senfina_ripeto::create_from( 1, 0.f, 100.f, 0.f )
        //);
        //PrimitivesH.push_back(
        //    primitives::senfina_ripeto::create_from( 1, 100.f, 0.f, 0.f )
        //);
        //PrimitivesH.push_back(
        //    primitives::sfero::create_from( 65.f )
        //);

        float3 d{ 1.f, 30.f, 1.f };
        float theta = -1.8f, w = cosf( theta / 2.f ), r = sinf( theta / 2.f ) / sqrtf( d.x * d.x + d.y * d.y + d.z * d.z );

        // TUBARETKA
        //PrimitivesH.push_back(
        //    primitives::senfina_ripeto::create_from( 1, 0.f, 500.f, 100.f )
        //);
        //PrimitivesH.push_back(
        //    primitives::movo::create_from( 1, 200.f, 0.f, 0.f )
        //);
        //PrimitivesH.push_back(
        //    primitives::rotacioQ::create_from( 1, w, r * d.x, r * d.y, r * d.z )
        //);
        //PrimitivesH.push_back(
        //    primitives::komunajo_2::create_from( 1, 2 )
        //);
        //PrimitivesH.push_back(
        //    primitives::kubo::create_from( 50.f, 50.f, 50.f )
        //);
        //PrimitivesH.push_back(
        //    primitives::komplemento::create_from( 1 )
        //);
        //PrimitivesH.push_back(
        //    primitives::kunigajo_2::create_from( 1, 3 )
        //);
        //PrimitivesH.push_back(
        //    primitives::movo::create_from( 1, 0.f, 0.f, -50.f )
        //);
        //PrimitivesH.push_back(
        //    primitives::sfero::create_from( 60.f )
        //);
        //PrimitivesH.push_back(
        //    primitives::movo::create_from( 1, 0.f, 0.f, 50.f )
        //);
        //PrimitivesH.push_back(
        //    primitives::sfero::create_from( 40.f )
        //);

        // PLANE
        //PrimitivesH.push_back(
        //    primitives::kunigajo_2::create_from( 1, 2 )
        //);
        //PrimitivesH.push_back(
        //    primitives::ebeno::create_from( 0.f, 0.f, 1.f )
        //);
        //PrimitivesH.push_back(
        //    primitives::movo::create_from( 1, 0.f, 0.f, 50.f )
        //);
        //PrimitivesH.push_back(
        //    primitives::kubo::create_from( 50.f, 50.f, 50.f )
        //);
    }*/
    point d{ 1.f, 30.f, 1.f };
    float alpha = -1.8f, w = cosf( alpha / 2.f ), r = sinf( alpha / 2.f ) / sqrtf( d.x * d.x + d.y * d.y + d.z * d.z );
    
    CUDA->ReservePrimitives( RAYS_BLOCK_2D_x * RAYS_BLOCK_2D_y * RAYS_PRIMITIVES_PER_THREAD );
    CUDA->AddPrimitive( cuda::senfina_ripeto::create_from( 1, 0.f, 500.f, 100.f ) );
    CUDA->AddPrimitive( cuda::movo::create_from( 1, 200.f, 0.f, 0.f ) );
    CUDA->AddPrimitive( cuda::rotacioQ::create_from( 1, w, r * d.x, r * d.y, r * d.z ) );
    CUDA->AddPrimitive( cuda::komunajo_2::create_from( 1, 2 ) );
    CUDA->AddPrimitive( cuda::kubo::create_from( 50.f, 50.f, 50.f ) );
    CUDA->AddPrimitive( cuda::komplemento::create_from( 1 ) );
    CUDA->AddPrimitive( cuda::kunigajo_2::create_from( 1, 3 ) );
    CUDA->AddPrimitive( cuda::movo::create_from( 1, 0.f, 0.f, -50.f ) );
    CUDA->AddPrimitive( cuda::sfero::create_from( 60.f ) );
    CUDA->AddPrimitive( cuda::movo::create_from( 1, 0.f, 0.f, 50.f ) );
    CUDA->AddPrimitive( cuda::sfero::create_from( 40.f ) );
    CUDA->SetPrimitives();

    float scale = powf( 2.f, -6.1f ), theta = 0.f, cos_theta = 1.f, sin_theta = 0.f, phi = 0.f, cos_phi = 1.f, sin_phi = 0.f;
    float cos_1 = cosf( M_PI / 180.f ), sin_1 = sinf( M_PI / 180.f ), deg = M_PI / 360.f;
    
    rays_info Info_h = {
            .Depth = 1000,
            .LightSource = point{ -2.f * M_SQRT1_5f, 0.f, M_SQRT1_5f },
            .StartPos = point{ 0.f, 0.f, 0.f },
            .StartDir = point{ scale * cos_theta * cos_phi, scale * cos_theta * sin_phi, scale * sin_theta },
            .StartWVec = point{ scale * sin_phi, -scale * cos_phi, 0.f },
            .StartHVec = point{ scale * sin_theta * cos_phi, scale * sin_theta * sin_phi, -scale * cos_theta }
    };
    
    CUDA->SetInfo( Info_h );
    
    size_t compute_time = 0, this_time, prev_time;
    bool MIDDLE_BUTTON = false, RIGHT_BUTTON = false, run;
    for ( run = true, this_time = SDL_GetTicks(), prev_time = this_time;
          run;
          prev_time = this_time, this_time = SDL_GetTicks() ) {
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
                    Info_h.LightSource = point{
                        cos_1 * Info_h.LightSource.x - sin_1 * Info_h.LightSource.y,
                        sin_1 * Info_h.LightSource.x + cos_1 * Info_h.LightSource.y,
                        Info_h.LightSource.z
                    };
                    break;
                case SDLK_RIGHT:
                    Info_h.LightSource = point{
                        cos_1 * Info_h.LightSource.x + sin_1 * Info_h.LightSource.y,
                        -sin_1 * Info_h.LightSource.x + cos_1 * Info_h.LightSource.y,
                        Info_h.LightSource.z
                    };
                    break;
                case SDLK_SPACE:
                    scale = powf( 2.f, -2.1f );
                    theta = 0.f;
                    cos_theta = 1.f;
                    sin_theta = 0.f;
                    phi = 0.f;
                    cos_phi = 1.f;
                    sin_phi = 0.f;

                    Info_h.StartPos = point{ 0.f, 0.f, 0.f };
                    Info_h.StartDir = point{ scale * cos_theta * cos_phi, scale * cos_theta * sin_phi, scale * sin_theta };
                    Info_h.StartWVec = point{ scale * sin_phi, -scale * cos_phi, 0.f };
                    Info_h.StartHVec = point{ scale * sin_theta * cos_phi, scale * sin_theta * sin_phi, -scale * cos_theta };

                    break;
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

                Info_h.StartDir = point{ scale * cos_theta * cos_phi, scale * cos_theta * sin_phi, scale * sin_theta };
                Info_h.StartWVec = point{ scale * sin_phi, -scale * cos_phi, 0.f };
                Info_h.StartHVec = point{ scale * sin_theta * cos_phi, scale * sin_theta * sin_phi, -scale * cos_theta };

                break;
            case SDL_MOUSEMOTION:
                prevMouse = currMouse;
                currMouse = SDL_Point{ Event.motion.x, Event.motion.y };

                if ( MIDDLE_BUTTON ) {
                    float S = 2.f / scale;
                    Info_h.StartPos.x -= float( currMouse.x - prevMouse.x ) * Info_h.StartWVec.x * S;
                    Info_h.StartPos.y -= float( currMouse.x - prevMouse.x ) * Info_h.StartWVec.y * S;
                    Info_h.StartPos.z -= float( currMouse.x - prevMouse.x ) * Info_h.StartWVec.z * S;
                    Info_h.StartPos.x -= float( currMouse.y - prevMouse.y ) * Info_h.StartHVec.x * S;
                    Info_h.StartPos.y -= float( currMouse.y - prevMouse.y ) * Info_h.StartHVec.y * S;
                    Info_h.StartPos.z -= float( currMouse.y - prevMouse.y ) * Info_h.StartHVec.z * S;
                } else if ( RIGHT_BUTTON ) {
                    theta += float( currMouse.y - prevMouse.y ) * deg;
                    if ( theta > M_PI_2f )
                        theta = M_PI_2f;
                    else if ( theta < -M_PI_2f )
                        theta = -M_PI_2f;
                    cos_theta = cosf( theta );
                    sin_theta = sinf( theta );

                    phi += float( currMouse.x - prevMouse.x ) * deg;
                    cos_phi = cosf( phi );
                    sin_phi = sinf( phi );
                }

                Info_h.StartDir = point{ scale * cos_theta * cos_phi, scale * cos_theta * sin_phi, scale * sin_theta };
                Info_h.StartWVec = point{ scale * sin_phi, -scale * cos_phi, 0.f };
                Info_h.StartHVec = point{ scale * sin_theta * cos_phi, scale * sin_theta * sin_phi, -scale * cos_theta };

                break;
            }
        }

        Info_h.LightSource = point{
            cos_1 * Info_h.LightSource.x - sin_1 * Info_h.LightSource.y,
            sin_1 * Info_h.LightSource.x + cos_1 * Info_h.LightSource.y,
            Info_h.LightSource.z
        };

        // Render Scene
        CUDA->SetInfo( Info_h );
        compute_time = CUDA->Process();

        // Draw Scene
        glBegin( GL_QUADS ); {
            glTexCoord2i( 0, 0 );
            glVertex2i( 0, 0 );
            glTexCoord2i( 1, 0 );
            glVertex2i( Width, 0 );
            glTexCoord2i( 1, 1 );
            glVertex2i( Width, Height );
            glTexCoord2i( 0, 1 );
            glVertex2i( 0, Height );
        } glEnd();

        // Scene update
        SDL_GL_SwapWindow( Win );
#ifndef _DEBUG
        cout << "\rFrame time : " << size_t( this_time - prev_time ) << "ms  \t\tTask execution time : " << compute_time << "ms  " << flush;
#endif
    }


    CUDA->Quit();

    SDL_GL_DeleteContext( GL );
    SDL_DestroyWindow( Win );

    SDL_Quit();

    return 0;
}