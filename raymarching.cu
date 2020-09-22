#include "rays.h"
#include <cuda_device_runtime_api.h>

namespace primitives {
//__device__ scalar sphere::dist( byte *data, const point& p ) {
//    point c = DATA->c;
//    return norm3df( p.x - c.x, p.y - c.y, p.z - c.z ) - DATA->r;
//}
//
//__device__ point sphere::norm( byte *data, const point& p ) {
//    point c = DATA->c, dp = point{ p.x - c.x, p.y - c.y , p.z - c.z };
//    return mul_point( dp, rnorm3df( dp.x, dp.y, dp.z ) );
//}
//
//__host__ base_ptr sphere::create( const point& center, const scalar& radius ) {
//    base_ptr NEW = new base( type_sphere );
//
//    byte_cast< data_struct > _data;
//    _data.data.r = radius;
//    _data.data.c = center;
//
//    memcpy( NEW->data, _data.source, sizeof data_struct );
//    return NEW;
//}

// SPHERE
CREATE_OBJECT_TYPE_DEFINITION( sphere,
                               {
                                   return norm3df( p.x - data->c.x, p.y - data->c.y, p.z - data->c.z ) - data->r;
                               },
                               {
                                   point dp; dp.x = p.x - data->c.x; dp.y = p.y - data->c.y; dp.z = p.z - data->c.z; return mul_point( dp, rnorm3df( dp.x, dp.y, dp.z ) );
                               } )

};

namespace raymarching {

static size_t Width, Height;
static cudaSurfaceObject_t Surface_d;

static point *LightSource_d;
static primitives::base *Primitives_d;
static size_t PrimitivesNum;
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
    int64_t
        x = RAYS_COORD_nD(x,2),
        y = RAYS_COORD_nD(y,2);

    if ( x < Info->Width && y < Info->Height ) {
        scalar
            X = .5f * ( 2 * x - int64_t( Info->Width ) + 1 ),
            Y = .5f * ( 2 * y - int64_t( Info->Height ) + 1 ),
            Z = Info->Depth;

        point pos;
        pos.x = X * Info->StartWVec.x + Y * Info->StartHVec.x + Z * Info->StartDir.x;
        pos.y = X * Info->StartWVec.y + Y * Info->StartHVec.y + Z * Info->StartDir.y;
        pos.z = X * Info->StartWVec.z + Y * Info->StartHVec.z + Z * Info->StartDir.z;

        scalar R_1 = rnorm3df( pos.x, pos.y, pos.z );

        ray *self = Rays + y * Info->Width + x;
        self->d = Info->StartDir;//mul_point( pos, R_1 );
        self->p = add_point( pos, Info->StartPos );
    }
}

int Start( point &LightSource, std::list< primitives::base_ptr > &Primitives, const size_t &width, const size_t &height, const cudaSurfaceObject_t &surface, cudaStream_t stream ) {
    Width = width;
    Height = height;
    Surface_d = surface;

    cudaMalloc( &LightSource_d, sizeof point );
    cudaMemcpyAsync( LightSource_d, &LightSource, sizeof point, cudaMemcpyHostToDevice, stream );

    PrimitivesNum = Primitives.size();
    size_t i = 0;
    cudaMalloc( &Primitives_d, PrimitivesNum * sizeof primitives::base );
    for ( primitives::base_ptr ptr : Primitives ) {
        //ptr = primitives::sphere::create( point{ 200.f, 0.f, 0.f }, 50.f );
        cudaMemcpyAsync( Primitives_d + i, ptr, sizeof primitives::base, cudaMemcpyHostToDevice, stream );
        ++i;
    }

    cudaMalloc( &Rays_d, Width * Height * sizeof ray );

    start_init_rays_info Info_h;
    Info_h.Width = Width;
    Info_h.Height = Height;
    Info_h.Depth = 100; // max( Width, Height );
    Info_h.StartPos = point{ 0.f, 0.f, 0.f };
    Info_h.StartDir = point{ 1.f, 0.f, 0.f };
    Info_h.StartWVec = point{ 0.f, -1.f, 0.f };
    Info_h.StartHVec = point{ 0.f, 0.f, -1.f };
    cudaMalloc( &Info_d, sizeof start_init_rays_info );
    cudaMemcpyAsync( Info_d, &Info_h, sizeof start_init_rays_info, cudaMemcpyHostToDevice, stream );

    kernelStart <<< grid( Width, Height ), block_2d, 0, stream >>> ( Info_d, Rays_d );
    cudaStreamSynchronize( stream );

    return 1;
}

static __global__ void kernelImageProcessing( cudaSurfaceObject_t image, size_t width, size_t height, size_t time, ray KERNEL_PTR Rays, point KERNEL_PTR LightSource, primitives::base KERNEL_PTR Primitives, size_t PrimitivesNum ) {
    size_t  x = RAYS_COORD_nD( x, 2 ),
            y = RAYS_COORD_nD( y, 2 );

    if ( x < width && y < height ) {
        size_t I;
        scalar min_dist, curr_dist, ray_dist = 0, R = 50.f;
        point C = point{ 200.f, 0.f, 0.f };
        ray r = Rays[ y * width + x ];

        primitives::base    curr_object;
        size_t              curr_type;

        while ( ray_dist < RAYS_MAX_DIST ) {
            min_dist = RAYS_MAX_DIST;

            for ( size_t I = 0; I < PrimitivesNum; ++I ) {
                curr_object = Primitives[ I ];
                curr_type = curr_object.type;

                switch ( curr_type ) {
                CREATE_OBJECT_TYPE_PROCESSING( sphere )
                default:
                    curr_dist = RAYS_MAX_DIST;
                }

                if ( min_dist > curr_dist )
                    min_dist = curr_dist;
            }

            r.p.x += min_dist * r.d.x;
            r.p.y += min_dist * r.d.y;
            r.p.z += min_dist * r.d.z;

            if ( min_dist < RAYS_MIN_DIST ) {
                uchar3 COLOR = uchar3{ 0xff, 0xff, 0xff };
                uchar4 PIXEL = RGB_PIXEL( COLOR );
                surf2Dwrite( PIXEL, image, x * 4, y );
                break;
            }

            ray_dist += min_dist;
        }
    }
}

bool ImageProcessing( size_t time, cudaStream_t stream ) {
    kernelImageProcessing <<< grid( Width, Height ), block_2d, 0, stream >>>
        ( Surface_d, Width, Height, time, Rays_d, LightSource_d, Primitives_d, PrimitivesNum );
    cudaStreamSynchronize( stream );
    return true;
}

};