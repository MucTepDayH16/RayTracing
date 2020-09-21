#include "rays.h"
#include <cuda_device_runtime_api.h>

#define RAYS_BLOCK_X 16
#define RAYS_BLOCK_Y 16
#define RAYS_BLOCK_Z 16

namespace raymarching {

static dim3 block_1d( RAYS_BLOCK_X );
static dim3 block_2d( RAYS_BLOCK_X, RAYS_BLOCK_Y );
static dim3 block_3d( RAYS_BLOCK_X, RAYS_BLOCK_Y, RAYS_BLOCK_Z );

static dim3 grid( size_t X ) {
    return dim3( ( X - 1 ) / RAYS_BLOCK_X + 1 );
}

static dim3 grid( size_t X, size_t Y ) {
    return dim3( ( X - 1 ) / RAYS_BLOCK_X + 1, ( Y - 1 ) / RAYS_BLOCK_Y + 1 );
}

static dim3 grid( size_t X, size_t Y, size_t Z ) {
    return dim3( ( X - 1 ) / RAYS_BLOCK_X + 1, ( Y - 1 ) / RAYS_BLOCK_Y + 1, ( Z - 1 ) / RAYS_BLOCK_Z + 1 );
}

static __device__ __inline__ point mul_point( const point& p, const scalar& s ) {
    return point{ s * p.x, s * p.y, s * p.z };
}

static __device__ __inline__ point add_point( const point& p1, const point& p2 ) {
    return point{ p1.x + p2.x, p1.y + p2.y, p1.z + p2.z };
}

static __global__ void start_init_rays( start_init_rays_info *__restrict__ Info, ray *__restrict__ Rays ) {
    size_t
        x = blockIdx.x * RAYS_BLOCK_X + threadIdx.x,
        y = blockIdx.y * RAYS_BLOCK_Y + threadIdx.y;

    if ( x < Info->Width && y < Info->Height ) {
        scalar
            X = .5f * ( 2 * x - Info->Width + 1 ),
            Y = .5f * ( 2 * y - Info->Height + 1 ),
            Z = Info->Depth;

        point pos;
        pos.x = X * Info->StartWVec.x + Y * Info->StartHVec.x + Z * Info->StartDir.x;
        pos.y = X * Info->StartWVec.y + Y * Info->StartHVec.y + Z * Info->StartDir.y;
        pos.z = X * Info->StartWVec.z + Y * Info->StartHVec.z + Z * Info->StartDir.z;

        scalar R_1 = rnorm3df( pos.x, pos.y, pos.z );

        ray *self = Rays + y * Info->Width + x;
        self->d = mul_point( pos, R_1 );
        self->p = add_point( pos, Info->StartPos );
    }
}

int start( point * LightSource, primitives::base * Primitives, const size_t Width, const size_t Height ) {
    ray *RaysD;
    cudaMalloc( &RaysD, Width * Height * sizeof ray );

    start_init_rays_info InfoH, *InfoD;
    InfoH.Width = Width;
    InfoH.Height = Height;
    InfoH.Depth = max( Width, Height );
    InfoH.StartPos = point{ 0.f, 0.f, 0.f };
    InfoH.StartDir = point{ 1.f, 0.f, 0.f };
    InfoH.StartWVec = point{ 0.f, -1.f, 0.f };
    InfoH.StartHVec = point{ 0.f, 0.f, -1.f };
    cudaMalloc( &InfoD, sizeof start_init_rays_info );
    cudaMemcpy( InfoD, &InfoH, sizeof start_init_rays_info, cudaMemcpyHostToDevice );

    start_init_rays << < grid( Width, Height ), block_2d, 0, 0 >> > ( InfoD, RaysD );

    return 1;
}

};