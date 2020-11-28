#pragma once

#include "rays.h"
#include "cuda_defines.cuh"

#include <cuda.h>
#include <cudaGL.h>
#include <nvrtc.h>

namespace cuda {

class raymarching : public null::raymarching {
    CUdevice                _device;
    uint64_t                _cc_div_10;
    char                    _device_name[128];
    
    CUcontext               _context;
    CUmodule                _module;
    CUlinkState             _link_state;
    CUstream                _stream[ CUDA_RAYS_STREAM_NUM ], _default_stream;
    CUevent                 _event[ CUDA_RAYS_EVENT_NUM ];
    
    float                   _last_process_time;
    CUresult                _last_cuda_error;
    nvrtcResult             _last_nvrtc_error;
    
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

};