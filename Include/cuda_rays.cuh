#pragma once

#include "rays.h"
#include <cuda.h>

#define CUDA_RAYS_STREAM_NUM        1
#define CUDA_RAYS_DEFAULT_STREAM    0

#define CUDA_RAYS_EVENT_NUM         2

#ifdef _DEBUG
#define CUDA_CHECK(__ERROR__) {                                                         \
    CUresult err = ( __ERROR__ );                                                       \
    if ( err ) {                                                                        \
        const char *ErrorName;                                                          \
        cuGetErrorName(__ERROR__,&ErrorName);                                           \
        std::cout <<                                                                    \
            ErrorName << std::endl <<                                                   \
            "\t: at line " << __LINE__ << std::endl <<                                  \
            "\t: in file " << __FILE__ << std::endl << std::endl;                       \
    }                                                                                   \
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
    CUdevice                _device;
    uint8_t                 _cc_div_10;
    char                    _device_name[128];
    
    CUcontext               _context;
    CUmodule                _module;
    CUstream                _stream[ CUDA_RAYS_STREAM_NUM ], _default_stream;
    CUevent                 _event[ CUDA_RAYS_EVENT_NUM ];
    
    float                   _last_process_time;
    CUresult                _last_cuda_error;
    
    CUfunction              _process, _set_rays, _set_primitives;
    CUdeviceptr             _rays, _info, _prim;
    CUdeviceptr             _width, _height, _prim_num;
    
    CUgraphicsResource      _resource;
    CUDA_RESOURCE_DESC      _resource_desc;
    CUsurfObject            _surface;
public:
    int Init( rays_Init_args ) override;
    int Process( rays_Process_args ) override;
    int Quit( rays_Quit_args ) override;
    
    int SetInfo( rays_SetInfo_args ) override;
    int SetTexture( rays_SetTexture_args ) override;
    int UnsetTexture( rays_UnsetInfo_args ) override;
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