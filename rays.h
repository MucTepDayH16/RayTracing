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

struct bazo;
typedef bazo* bazo_ptr;

typedef scalar( *dist_func )( bazo_ptr, const point& );
typedef point( *norm_func )( bazo_ptr, const point& );

enum object_type {
    type_nenio = 0x0000,
    type_sfero,
    type_kubo,
    type_cilindro,

    type_kunigajo_2 = 0x0100,
    type_kunigajo_3,
    type_kunigajo_4,
    type_komunajo_2,
    type_komunajo_3,
    type_komunajo_4,
    type_komplemento,
    type_glata_kunigajo_2,
    type_glata_komunajo_2,

    type_movo = 0x0200,
    type_rotacioX,
    type_rotacioY,
    type_rotacioZ,
    type_rotacioQ
};

static __device__ __inline__ point mul_point( const point& p, const scalar& s ) {
    return point{ s * p.x, s * p.y, s * p.z };
}

template< typename __TYPE__ >
union byte_cast {
    __TYPE__ data;
    byte source[ sizeof __TYPE__ ];
};

static const size_t bazo_payload_size = 24;
struct bazo {
    byte data[ bazo_payload_size ];
    dist_func dist;
    norm_func norm;
    enum object_type type;
    __device__ __host__ bazo( enum object_type _type = type_nenio ) : type( _type ), dist( nullptr ), norm( nullptr ) {}
};

// TYPE_LIST
CREATE_OBJECT_TYPE_DESCRIPTION( sfero, struct { scalar r; } )
CREATE_OBJECT_TYPE_DESCRIPTION( kubo, struct { point b; } )
CREATE_OBJECT_TYPE_DESCRIPTION( cilindro, struct { scalar r; scalar h; } )

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

};

namespace raymarching {

typedef unsigned char byte;
typedef float scalar;
typedef float3 point;
typedef struct { point p, d; } ray;
typedef struct { size_t Width, Height, Depth; point StartPos, StartDir, StartWVec, StartHVec; } start_init_rays_info;

int Init( size_t, size_t, size_t, const cudaSurfaceObject_t& );
int Load( point &LightSource, std::list< primitives::bazo_ptr > &Primitives, start_init_rays_info &Info, cudaStream_t stream = 0 );
bool ImageProcessing( size_t, cudaStream_t stream = 0 );
bool Quit();

};