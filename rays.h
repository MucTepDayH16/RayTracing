#pragma once

#include <list>
#include <cmath>
#include <memory>
#include <bitset>
#include <iomanip>
#include <iostream>

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

typedef unsigned char byte;
typedef long counter;
typedef float scalar;
typedef float3 point;
typedef struct { point x, y, z; } matrix;
typedef float4 point4;
typedef struct { point4 x, y, z, w; } matrix4;

struct base;
typedef base* base_ptr;

typedef scalar( *dist_func )( base_ptr, const point& );
typedef point( *norm_func )( base_ptr, const point& );

enum object_type {
    type_none = 0x0000,
    type_sphere,
    type_cube,

    type_unification = 0x0100,
    type_intersection,
    type_invertion,
    type_smooth_unification,
    type_smooth_intersection,
    type_smooth_invertion,

    type_translation = 0x0200,
    type_rotationX,
    type_rotationY,
    type_rotationZ,
    type_rotationQ
};

static __device__ __inline__ point mul_point( const point& p, const scalar& s ) {
    return point{ s * p.x, s * p.y, s * p.z };
}

template< typename __TYPE__ >
union byte_cast {
    __TYPE__ data;
    byte source[ sizeof __TYPE__ ];
};

static const size_t base_payload_size = 24;
struct base {
    byte data[ base_payload_size ];
    dist_func dist;
    norm_func norm;
    enum object_type type;
    __device__ __host__ base( enum object_type _type = type_none ) : type( _type ), dist( nullptr ), norm( nullptr ) {}
};

// TYPE_LIST
CREATE_OBJECT_TYPE_DESCRIPTION( sphere, struct { scalar r; } )
CREATE_OBJECT_TYPE_DESCRIPTION( cube, struct { point b; } )

CREATE_OBJECT_TYPE_DESCRIPTION( unification, struct { counter o1; counter o2; } )
CREATE_OBJECT_TYPE_DESCRIPTION( intersection, struct { counter o1; counter o2; } )
CREATE_OBJECT_TYPE_DESCRIPTION( invertion, struct { counter o; } )
CREATE_OBJECT_TYPE_DESCRIPTION( smooth_unification, struct { counter o1; counter o2; scalar k; } )
CREATE_OBJECT_TYPE_DESCRIPTION( smooth_intersection, struct { counter o1; counter o2; scalar k; } )

CREATE_OBJECT_TYPE_DESCRIPTION( translation, struct { counter o; point t; } )
CREATE_OBJECT_TYPE_DESCRIPTION( rotationX, struct { counter o; scalar cos_phi; scalar sin_phi; } )
CREATE_OBJECT_TYPE_DESCRIPTION( rotationY, struct { counter o; scalar cos_phi; scalar sin_phi; } )
CREATE_OBJECT_TYPE_DESCRIPTION( rotationZ, struct { counter o; scalar cos_phi; scalar sin_phi; } )
CREATE_OBJECT_TYPE_DESCRIPTION( rotationQ, struct { counter o; scalar q_w; point q; } )

};

namespace raymarching {

typedef unsigned char byte;
typedef float scalar;
typedef float3 point;
typedef struct { point p, d; } ray;
typedef struct { size_t Width, Height, Depth; point StartPos, StartDir, StartWVec, StartHVec; } start_init_rays_info;

int Init( size_t, size_t, size_t, const cudaSurfaceObject_t& );
int Load( point &LightSource, std::list< primitives::base_ptr > &Primitives, start_init_rays_info &Info, cudaStream_t stream = 0 );
bool ImageProcessing( size_t, cudaStream_t stream = 0 );
bool Quit();

};