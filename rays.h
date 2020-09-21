#pragma once
#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>

namespace cuda {

template< typename __TYPE__ >
class pointer {
protected:
    static const size_t size_of_type = sizeof __TYPE__;
    typedef __TYPE__* __POINTER__;

    size_t elem_num;
    __POINTER__ device_ptr;
public:
    pointer( size_t num = 1 ) : elem_num( num ), device_ptr( nullptr ) {
        cudaMalloc( &device_ptr, elem_num * size_of_type );
    }
    ~pointer() {
        cudaFree( device_ptr );
    }

    int CopyFrom( __POINTER__ host_ptr, size_t offset = 0, size_t num = 1 ) {
        return cudaMemcpyAsync( device_ptr + offset, host_ptr, num * size_of_type, cudaMemcpyHostToDevice );
    }
    int CopyFrom( const pointer<__TYPE__> &other_ptr, size_t offset = 0, size_t other_offset = 0, size_t num = 1 ) {
        return cudaMemcpyAsync( device_ptr + offset, other_ptr.device_ptr + other_offset, num * size_of_type, cudaMemcpyDeviceToDevice );
    }

    int CopyTo( __POINTER__ host_ptr, size_t offset = 0, size_t num = 1 ) {
        return cudaMemcpyAsync( host_ptr, device_ptr + offset, num * size_of_type, cudaMemcpyDeviceToHost );
    }
    int CopyTo( const pointer<__TYPE__> &other_ptr, size_t offset = 0, size_t other_offset = 0, size_t num = 1 ) {
        return cudaMemcpyAsync( other_ptr.device_ptr + other_offset, device_ptr + offset, num * size_of_type, cudaMemcpyDeviceToDevice );
    }

    operator __POINTER__() { return device_ptr; }
};

};

namespace primitives {

typedef unsigned char byte;
typedef float scalar;
typedef float3 point;

class base {
protected:
    char data[ 32 ];
public:
    virtual __device__ scalar dist( const point& ) = 0;
    virtual __device__ point norm( const point& ) = 0;
};

typedef base* base_ptr;

class sphere : public base {
protected:
// data = radius(4) , center_x(4) , center_y(4) , center_z(4)
    typedef struct alignas(4) { scalar r; point c; } data_struct;
public:
    sphere( const point&, const scalar& );

    __device__ scalar dist( const point& ) override;
    __device__ point norm( const point& ) override;
};

};

namespace raymarching {

typedef unsigned char byte;
typedef float scalar;
typedef float3 point;
typedef struct { point p, d; } ray;
typedef struct { size_t Width, Height, Depth; point StartPos, StartDir, StartWVec, StartHVec; } start_init_rays_info;

int start( point *LightSource, primitives::base *Primitives, const size_t Width, const size_t Height );

};