#pragma once
#include "rays.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define CUDA_RAYS_STREAM_NUM        1
#define CUDA_RAYS_DEFAULT_STREAM    0

#define CUDA_RAYS_EVENT_NUM         2

#define CUDA_SET_GRID(c,n) ( c - 1 ) / Block##n##d.c + 1
#ifdef _DEBUG
#define CUDA_CHECK(__ERROR__) {                                                         \
    cudaError_t err = ( __ERROR__ );                                                    \
    if ( err )                                                                          \
        std::cout <<                                                                    \
            cudaGetErrorName( __ERROR__ ) << std::endl <<                               \
            "\t: at line " << __LINE__ << std::endl <<                                  \
            "\t: in file " << __FILE__ << std::endl << std::endl;                       \
}
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(__ERROR__) (void)__ERROR__
#endif

#define length_2(x,y)       (hypotf((x),(y)))
#define length_3(x,y,z)     (norm3df((x),(y),(z)))

#define r_length_2(x,y)     (rhypotf((x),(y)))
#define r_length_3(x,y,z)   (rnorm3df((x),(y),(z)))

#define CUDA_RAYS_COORD_nD(c,n)  blockIdx.##c * RAYS_BLOCK_##n##D_##c + threadIdx.##c

namespace cuda {

class raymarching : public null::raymarching {
    cudaStream_t            _stream[ CUDA_RAYS_STREAM_NUM ], _default_stream;
    cudaEvent_t             _event[ CUDA_RAYS_EVENT_NUM ];
    float                   _last_process_time;
    cudaError_t             _last_cuda_error;
    cudaGraphicsResource*   _resource;
    cudaResourceDesc        _resource_desc;
    cudaSurfaceObject_t     _surface;
public:
    int Init( rays_Init_args ) override;
    int Process( rays_Process_args ) override;
    int Quit( rays_Quit_args ) override;
    
    int SetInfo( rays_SetInfo_args ) override;
    int SetTexture( rays_SetTexture_args ) override;
    int SetPrimitives( rays_SetPrimitives_args ) override;
    int SetRays( rays_SetRays_args ) override;
};

// TYPE_LIST
CREATE_OBJECT_TYPE_DESCRIPTION( portanta_sfero, struct { counter o; point t; scalar r; } )
CREATE_OBJECT_TYPE_DESCRIPTION( sfero, struct { scalar r; } )
CREATE_OBJECT_TYPE_DESCRIPTION( kubo, struct { point b; } )
CREATE_OBJECT_TYPE_DESCRIPTION( cilindro, struct { scalar r; scalar h; } )

CREATE_OBJECT_TYPE_DESCRIPTION( ebeno, struct { point n; } )

CREATE_OBJECT_TYPE_DESCRIPTION( kunigajo_2, struct { counter o[ 2 ]; } )
CREATE_OBJECT_TYPE_DESCRIPTION( kunigajo_3, struct { counter o[ 3 ]; } )
CREATE_OBJECT_TYPE_DESCRIPTION( kunigajo_4, struct { counter o[ 4 ]; } )
CREATE_OBJECT_TYPE_DESCRIPTION( komunajo_2, struct { counter o[ 2 ]; } )
CREATE_OBJECT_TYPE_DESCRIPTION( komunajo_3, struct { counter o[ 3 ]; } )
CREATE_OBJECT_TYPE_DESCRIPTION( komunajo_4, struct { counter o[ 4 ]; } )
CREATE_OBJECT_TYPE_DESCRIPTION( komplemento, struct { counter o; } )
CREATE_OBJECT_TYPE_DESCRIPTION( glata_kunigajo_2, struct { counter o[ 2 ]; scalar k; } )
CREATE_OBJECT_TYPE_DESCRIPTION( glata_komunajo_2, struct { counter o[ 2 ]; scalar k; } )

CREATE_OBJECT_TYPE_DESCRIPTION( movo, struct { counter o; point t; } )
CREATE_OBJECT_TYPE_DESCRIPTION( rotacioX, struct { counter o; scalar cos_phi; scalar sin_phi; } )
CREATE_OBJECT_TYPE_DESCRIPTION( rotacioY, struct { counter o; scalar cos_phi; scalar sin_phi; } )
CREATE_OBJECT_TYPE_DESCRIPTION( rotacioZ, struct { counter o; scalar cos_phi; scalar sin_phi; } )
CREATE_OBJECT_TYPE_DESCRIPTION( rotacioQ, struct { counter o; scalar q_w; point q; } )
CREATE_OBJECT_TYPE_DESCRIPTION( senfina_ripeto, struct { counter o; point a; } )

};