#include "../Include/cuda_rays.cuh"

#define _CUDA(__ERROR__)    {_last_cuda_error = CUresult(__ERROR__); CUDA_CHECK(_last_cuda_error);}
#define _RETURN             return int(_last_cuda_error);

#define grid_1d(X)          (dim3( ( (X) - 1 ) / RAYS_BLOCK_1D_x + 1 ))
#define grid_2d(X,Y)        (dim3( ( (X) - 1 ) / RAYS_BLOCK_2D_x + 1, ( (Y) - 1 ) / RAYS_BLOCK_2D_y + 1 ))

#define block_1d            (dim3((RAYS_BLOCK_1D_x)))
#define block_2d            (dim3((RAYS_BLOCK_2D_x),(RAYS_BLOCK_2D_y)))

#define _KERNEL(__FUNC__)    _CUDA(cuLaunchKernel((__FUNC__),grid.x,grid.y,grid.z,block.x,block.y,block.z,0,_default_stream,args,nullptr))

#ifdef _DEBUG
#define _PATH "./cmake-build-debug/CMakeFiles/CudaPTX.dir/Source/cuda_kernels.ptx"
#else
#define _PATH "./cmake-build-release/CMakeFiles/CudaPTX.dir/Source/cuda_kernels.ptx"
#endif

namespace cuda {

int raymarching::Init( rays_Init_args ) {
    _CUDA( cuInit( 0 ) )
    
    //  Only one device
    _CUDA( cuDeviceGet( &_device, 0 ) )
    int cc[2];
    _CUDA( cuDeviceComputeCapability( cc, cc + 1, _device ) )
    _cc_div_10 = 10 * cc[0] + cc[1];
    _CUDA( cuDeviceGetName( _device_name, 128, _device ) )
    
    _CUDA( cuCtxCreate_v2( &_context, 0, _device ) )
    _CUDA( cuModuleLoad( &_module, _PATH ) )
    _CUDA( cuModuleGetFunction( &_process,          _module, "kernel_Process" ) )
    _CUDA( cuModuleGetFunction( &_set_primitives,   _module, "kernel_SetPrimitives" ) )
    _CUDA( cuModuleGetFunction( &_set_rays,         _module, "kernel_SetRays" ) )
    
    for ( size_t n = 0; n < CUDA_RAYS_STREAM_NUM; ++n ) {
        _CUDA( cuStreamCreate( _stream + n, 0 ) )
    }
    _default_stream = _stream[ CUDA_RAYS_DEFAULT_STREAM ];
    
    for ( size_t n = 0; n < CUDA_RAYS_EVENT_NUM; ++n ) {
        _CUDA( cuEventCreate( _event + n, 0 ) )
    }
    
    Width = width;
    Height = height;
    
    _CUDA( cuMemAlloc_v2( &_rays, Width * Height * sizeof( ray ) ) )
    _CUDA( cuMemAlloc_v2( &_info, sizeof( rays_info ) ) )
    _prim = 0;
    
    _CUDA( cuMemAlloc_v2( &_width, sizeof( size_t ) ) )
    _CUDA( cuMemAlloc_v2( &_height, sizeof( size_t ) ) )
    _CUDA( cuMemAlloc_v2( &_prim_num, sizeof( size_t ) ) )
    
    _resource = nullptr;
    
    _resource_desc.flags = 0;
    _resource_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    
    _RETURN
}

int raymarching::Process( rays_Process_args ) {
    _CUDA( cuEventRecord( _event[0], _default_stream ) )
    
    static dim3 grid = grid_2d( Width, Height ), block = block_2d;
    static void* args[] = { &_width, &_height, &_info, &_rays, &_prim, &_prim_num, &_surface };
    _KERNEL( _process )
    
    //  kernel::Process <<< grid_2d( Width, Height ), block_2d, 0, _default_stream >>>
    //      ( Width, Height, _INFO, _RAYS, _PRIM, PrimitivesNum, _surface );
    
    _CUDA( cuEventRecord( _event[ 1 ], _default_stream ) )
    _CUDA( cuStreamSynchronize( _default_stream ) )
    _CUDA( cuEventElapsedTime( &_last_process_time, _event[ 0 ], _event[ 1 ] ) )
    
    return uint32_t( _last_process_time );
}

int raymarching::Quit( rays_Quit_args ) {
    if (_prim) _CUDA( cuMemFree_v2( _prim ) )
    if (_rays) _CUDA( cuMemFree_v2( _rays ) )
    if (_info) _CUDA( cuMemFree_v2( _info ) )
    
    _CUDA( cuModuleUnload( _module ) )
    _CUDA( cuCtxDestroy_v2( _context ) )
    
    _RETURN
}


int raymarching::SetInfo( rays_SetInfo_args ) {
    _CUDA( cuMemcpyHtoDAsync_v2( _info, &info, sizeof( rays_info ), _default_stream ) )
    _CUDA( cuMemcpyHtoDAsync_v2( _width, &Width, sizeof( size_t ), _default_stream ) )
    _CUDA( cuMemcpyHtoDAsync_v2( _height, &Height, sizeof( size_t ), _default_stream ) )
    _CUDA( SetRays() )
    _CUDA( cuStreamSynchronize( _default_stream ) )

    
    _RETURN
}

int raymarching::SetTexture( rays_SetTexture_args ) {
    if ( _resource ) {
        _CUDA( UnsetTexture() )
    }
    
    _CUDA( cuGraphicsGLRegisterImage( &_resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore ) )
    _CUDA( cuGraphicsMapResources( 1, &_resource, _default_stream ) )
    
    _CUDA( cuGraphicsSubResourceGetMappedArray( &_resource_desc.res.array.hArray, _resource, 0, 0 ) )
    
    _CUDA( cuSurfObjectCreate( &_surface, &_resource_desc ) )
    
    _RETURN
}

int raymarching::UnsetTexture( rays_UnsetInfo_args ) {
    _CUDA( cuSurfObjectDestroy( _surface ) )
    _CUDA( cuGraphicsUnmapResources( 1, &_resource, _default_stream ) )
    _CUDA( cuGraphicsUnregisterResource( _resource ) )
    
    _resource = nullptr;
    
    _RETURN
}

int raymarching::SetPrimitives( rays_SetPrimitives_args ) {
    PrimitivesNum = Primitives_h.size();
    
    if ( _prim ) {
        _CUDA( cuMemFree_v2( _prim ) )
    }
    
    _CUDA( cuMemAlloc_v2( &_prim, PrimitivesNum * sizeof( primitives::bazo )  ) )
    _CUDA( cuMemcpyHtoDAsync_v2( _prim, Primitives_h.data(), PrimitivesNum * sizeof( primitives::bazo ), _default_stream ) )
    _CUDA( cuMemcpyHtoDAsync_v2( _prim_num, &PrimitivesNum, sizeof( size_t ), _default_stream ) )
    
    static dim3 grid = grid_1d( PrimitivesNum ), block = block_1d;
    static void* args[] = { &_prim, &_prim_num };
    _KERNEL( _set_primitives )
    
    //  kernel::SetPrimitives <<< grid_1d( PrimitivesNum ), block_1d, 0, _default_stream >>>
    //      ( _PRIM, PrimitivesNum );
    
    _CUDA( cuStreamSynchronize( _default_stream ) )
    
    _RETURN
}

int raymarching::SetRays( rays_SetRays_args ) {
    static dim3 grid = grid_2d( Width, Height ), block = block_2d;
    static void* args[] = { &_width, &_height, &_info, &_rays };
    _KERNEL( _set_rays )
    
    //  kernel::SetRays <<< grid_2d( Width, Height ), block_2d, 0, _default_stream >>>
    //      ( Width, Height, _INFO, _RAYS );
    
    _RETURN
}

};