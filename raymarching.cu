#include "rays.h"
#include <cuda_device_runtime_api.h>

#define RAYS_BLOCK_1D_x 256

#define RAYS_BLOCK_2D_x 16
#define RAYS_BLOCK_2D_y 16

#define RAYS_BLOCK_3D_x 8
#define RAYS_BLOCK_3D_y 8
#define RAYS_BLOCK_3D_z 4

#define RAYS_COORD_nD(c,n) blockIdx.##c * RAYS_BLOCK_##n##D_##c + threadIdx.##c

#define KERNEL_PTR *__restrict__

namespace raymarching {

static point *LightSource_d;
static primitives::base *Primitives_d;
static ray *Rays_d;
static start_init_rays_info *Info_d;


static dim3 block_1d( RAYS_BLOCK_1D_x );
static dim3 block_2d( RAYS_BLOCK_2D_x, RAYS_BLOCK_2D_y );
static dim3 block_3d( RAYS_BLOCK_3D_x, RAYS_BLOCK_3D_y, RAYS_BLOCK_3D_z );

static dim3 grid( size_t X ) {
    return dim3( ( X - 1 ) / RAYS_BLOCK_1D_x + 1 );
}

static dim3 grid( size_t X, size_t Y ) {
    return dim3( ( X - 1 ) / RAYS_BLOCK_2D_x + 1, ( Y - 1 ) / RAYS_BLOCK_2D_y + 1 );
}

static dim3 grid( size_t X, size_t Y, size_t Z ) {
    return dim3( ( X - 1 ) / RAYS_BLOCK_3D_x + 1, ( Y - 1 ) / RAYS_BLOCK_3D_y + 1, ( Z - 1 ) / RAYS_BLOCK_3D_z + 1 );
}

static __device__ __inline__ point mul_point( const point& p, const scalar& s ) {
    return point{ s * p.x, s * p.y, s * p.z };
}

static __device__ __inline__ point add_point( const point& p1, const point& p2 ) {
    return point{ p1.x + p2.x, p1.y + p2.y, p1.z + p2.z };
}

static __global__ void kernelStart( start_init_rays_info KERNEL_PTR Info, ray KERNEL_PTR Rays ) {
    size_t
        x = RAYS_COORD_nD(x,2),
        y = RAYS_COORD_nD(y,2);

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

int Start( point *LightSource, primitives::base *Primitives, const size_t Width, const size_t Height, cudaStream_t stream ) {
    LightSource_d = LightSource;
    Primitives_d = Primitives;
    cudaMalloc( &Rays_d, Width * Height * sizeof ray );

    start_init_rays_info Info_h;
    Info_h.Width = Width;
    Info_h.Height = Height;
    Info_h.Depth = max( Width, Height );
    Info_h.StartPos = point{ 0.f, 0.f, 0.f };
    Info_h.StartDir = point{ 1.f, 0.f, 0.f };
    Info_h.StartWVec = point{ 0.f, -1.f, 0.f };
    Info_h.StartHVec = point{ 0.f, 0.f, -1.f };
    cudaMalloc( &Info_d, sizeof start_init_rays_info );
    cudaMemcpy( Info_d, &Info_h, sizeof start_init_rays_info, cudaMemcpyHostToDevice );

    kernelStart <<< grid( Width, Height ), block_2d, 0, stream >>> ( Info_d, Rays_d );
    cudaStreamSynchronize( stream );

    return 1;
}

static __global__ void kernelImageProcessing( cudaSurfaceObject_t image, size_t width, size_t height, size_t time ) {
    size_t  x = RAYS_COORD_nD( x, 2 ),
            y = RAYS_COORD_nD( y, 2 );

    if ( x < width && y < height ) {
        uchar4 pixel = uchar4{ time & 0xff, ( x + y ) & 0xff, 0x00, 0xff };
        surf2Dwrite( pixel, image, x << 2, y );
    }
}

bool ImageProcessing( cudaSurfaceObject_t image, size_t width, size_t height, size_t time, cudaStream_t stream ) {
    kernelImageProcessing <<< grid( width, height ), block_2d, 0, stream >>>
        ( image, width, height, time );
    cudaStreamSynchronize( stream );
    return true;
}

};