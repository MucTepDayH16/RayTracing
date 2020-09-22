#pragma once

#include <iostream>
#include <list>
#include <cmath>
#include <memory>

#include <SDL2\SDL.h>
#include <SDL2\SDL_image.h>
#include <SDL2\SDL_opengl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "defines.h"

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

bool Init( size_t StreamsCount = 1 );
cudaStream_t Stream( size_t );

};

namespace primitives {

static const size_t base_size = 24;

typedef unsigned char byte;
typedef float scalar;
typedef float3 point;

typedef scalar( *dist_func )( byte*, const point& );
typedef point( *norm_func )( byte*, const point& );

enum object_type {
    type_none,
    type_sphere
};

static __device__ __inline__ point mul_point( const point& p, const scalar& s ) {
    return point{ s * p.x, s * p.y, s * p.z };
}

template< typename __TYPE__ >
union byte_cast {
    __TYPE__ data;
    byte source[ sizeof __TYPE__ ];
};

struct base {
    byte data[ base_size ];
    enum object_type type;
    __device__ __host__ base( enum object_type _type = type_none ) : type( _type ) {}
};

template <enum object_type> class object : public base {};

typedef base* base_ptr;

// TYPE_LIST
CREATE_OBJECT_TYPE_DESCRIPTION( sphere, struct { point c; scalar r; } )

};

namespace raymarching {

typedef unsigned char byte;
typedef float scalar;
typedef float3 point;
typedef struct { point p, d; } ray;
typedef struct { size_t Width, Height, Depth; point StartPos, StartDir, StartWVec, StartHVec; } start_init_rays_info;

int Start( point &LightSource, std::list< primitives::base_ptr > &Primitives, const size_t &width, const size_t &height, const cudaSurfaceObject_t &surface, cudaStream_t stream = 0 );
bool ImageProcessing( size_t, cudaStream_t stream = 0 );

};