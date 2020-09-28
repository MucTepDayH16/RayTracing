#include "rays.h"
#include <cuda_device_runtime_api.h>

namespace primitives {

__device__ __forceinline__ point Point( scalar x, scalar y, scalar z ) {
    return point{ x, y, z };
}

__device__ __forceinline__ scalar mix( scalar a, scalar b, scalar x ) {
    return b + ( a - b ) * x;
}

__device__ __forceinline__ scalar dot( point a, point b ) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// TYPE_LIST
CREATE_OBJECT_TYPE_DEFINITION(
    portanta_sfero,
    {
        point P;
        P.x = p.x - data->t.x;
        P.y = p.y - data->t.y;
        P.z = p.z - data->t.z;
        scalar d = length_3( P.x, P.y, P.z ) - data->r;
        bazo_ptr o = obj + data->o;
        if ( d <= RAYS_MIN_DIST )   return RAYS_DIST( o, P );
        else                        return d;
    },
    {
        point P;
        P.x = p.x - data->t.x;
        P.y = p.y - data->t.y;
        P.z = p.z - data->t.z;
        bazo_ptr o = obj + data->o;
        return RAYS_NORM( o, P );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    sfero,
    {
        return length_3( p.x, p.y, p.z ) - data->r;
    },
    {
        return p;
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    kubo,
    {
        point q;
        q.x = fabsf( p.x ) - data->b.x;
        q.y = fabsf( p.y ) - data->b.y;
        q.z = fabsf( p.z ) - data->b.z;
        if ( q.x < 0.f && q.y < 0.f && q.z < 0.f )
            return max( q.x, max( q.y, q.z ) );
        else
            return length_3( max( q.x, 0.f ), max( q.y, 0.f ), max( q.z, 0.f ) );
    },
    {
        point q;
        q.x = fabsf( p.x ) - data->b.x;
        q.y = fabsf( p.y ) - data->b.y;
        q.z = fabsf( p.z ) - data->b.z;
        if ( q.x < 0.f && q.y < 0.f && q.z < 0.f )
            return q.x > q.z ? ( q.x > q.y ? Point( p.x > 0.f ? 1.f : -1.f, 0.f, 0.f ) : Point( 0.f, p.y > 0.f ? 1.f : -1.f, 0.f ) ) : ( q.y > q.z ? Point( 0.f, p.y > 0.f ? 1.f : -1.f, 0.f ) : Point( 0.f, 0.f, p.z > 0.f ? 1.f : -1.f ) );
        else
            return q.x > q.z ? ( q.x > q.y ? Point( p.x > 0.f ? 1.f : -1.f, 0.f, 0.f ) : Point( 0.f, p.y > 0.f ? 1.f : -1.f, 0.f ) ) : ( q.y > q.z ? Point( 0.f, p.y > 0.f ? 1.f : -1.f, 0.f ) : Point( 0.f, 0.f, p.z > 0.f ? 1.f : -1.f ) );
        //Point( q.x > 0.f ? ( p.x >= data->c.x ? 1.f : -1.f ) : 0.f, q.y > 0.f ? ( p.y >= data->c.y ? 1.f : -1.f ) : 0.f, q.z > 0.f ? ( p.z >= data->c.z ? 1.f : -1.f ) : 0.f );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    cilindro,
    {
        scalar r = length_2( p.x, p.y );
        float2 q;
        q.x = r - data->r;
        q.y = fabsf( p.z ) - data->h;
        if ( q.x < 0.f && q.y < 0.f )
            return q.x > q.y ? q.x : q.y;
        else
            return length_2( max( q.x, 0.f ), max( q.y, 0.f ) );
    },
    {
        scalar r = length_2( p.x, p.y );
        float2 q;
        q.x = r - data->r;
        q.y = fabsf( p.z ) - data->h;
        if ( q.x < 0.f && q.y < 0.f )
            return q.x > q.y ? Point( p.x, p.y, 0.f ) : Point( 0.f, 0.f, p.z > 0.f ? 1.f : -1.f );
        else
            return q.x > q.y ? Point( p.x, p.y, 0.f ) : Point( 0.f, 0.f, p.z > 0.f ? 1.f : -1.f );

    } );


CREATE_OBJECT_TYPE_DEFINITION(
    kunigajo_2,
    {
        bazo_ptr o0 = obj + data->o[ 0 ]; bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        return min( d0, d1 );
    },
    {
        bazo_ptr o0 = obj + data->o[ 0 ]; bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );

        if ( d0 < d1 )  return RAYS_NORM( o0, p );
        else            return RAYS_NORM( o1, p );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    kunigajo_3,
    {
        bazo_ptr o = obj + data->o[ 0 ];
        scalar d = RAYS_DIST( o, p );

        o = obj + data->o[ 1 ];
        d = min( d, RAYS_DIST( o, p ) );

        o = obj + data->o[ 2 ];
        return min( d, RAYS_DIST( o, p ) );
    },
    {
        counter i_min = 0;
        bazo_ptr o = obj + data->o[ 0 ];
        scalar d; scalar d_min = RAYS_DIST( o, p );

        o = obj + data->o[ 1 ];
        d = RAYS_DIST( o, p );
        if ( d_min > d ) { d_min = d; i_min = 1; }

        o = obj + data->o[ 2 ];
        d = RAYS_DIST( o, p );
        if ( d_min > d ) { i_min = 2; }

        o = obj + data->o[ i_min ];
        return RAYS_NORM( o, p );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    kunigajo_4,
    {
        bazo_ptr o0 = obj + data->o[ 0 ]; bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        return min( d0, d1 );
    },
    {
        bazo_ptr o0 = obj + data->o[ 0 ]; bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );

        if ( d0 < d1 )  return RAYS_NORM( o0, p );
        else            return RAYS_NORM( o1, p );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    komunajo_2,
    {
        bazo_ptr o0 = obj + data->o[ 0 ]; bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        return max( d0, d1 );
    },
    {
        bazo_ptr o0 = obj + data->o[ 0 ]; bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );

        if ( d0 > d1 )  return RAYS_NORM( o0, p );
        else            return RAYS_NORM( o1, p );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    komunajo_3,
    {
        bazo_ptr o = obj + data->o[ 0 ];
        scalar d = RAYS_DIST( o, p );

        o = obj + data->o[ 1 ];
        d = max( d, RAYS_DIST( o, p ) );

        o = obj + data->o[ 2 ];
        return max( d, RAYS_DIST( o, p ) );
    },
    {
        counter i_max = 0;
        bazo_ptr o = obj + data->o[ 0 ];
        scalar d; scalar d_max = RAYS_DIST( o, p );

        o = obj + data->o[ 1 ];
        d = RAYS_DIST( o, p );
        if ( d_max < d ) { d_max = d; i_max = 1; }

        o = obj + data->o[ 2 ];
        d = RAYS_DIST( o, p );
        if ( d_max < d ) { i_max = 2; }

        o = obj + data->o[ i_max ];
        return RAYS_NORM( o, p );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    komunajo_4,
    {
        bazo_ptr o0 = obj + data->o[ 0 ]; bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        return max( d0, d1 );
    },
    {
        bazo_ptr o0 = obj + data->o[ 0 ]; bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );

        if ( d0 > d1 )  return RAYS_NORM( o0, p );
        else            return RAYS_NORM( o1, p );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    komplemento,
    {
        bazo_ptr O = obj + data->o;
        scalar D = RAYS_DIST( O, p );
        return -D;
    },
    {
        bazo_ptr O = obj + data->o;
        point N = RAYS_NORM( O, p );
        return Point( -N.x, -N.y, -N.z );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    glata_kunigajo_2,
    {
        bazo_ptr o0 = obj + data->o[ 0 ]; bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        scalar h = ( 1.f - ( d0 - d1 ) / data->k ) * .5f;
        if ( h > 1.f )  return d0;
        if ( h < 0.f )  return d1;
        return mix( d0, d1, h ) - data->k * h * ( 1.f - h );
    },
    {
        bazo_ptr o0 = obj + data->o[ 0 ]; bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        scalar h = ( 1.f - ( d0 - d1 ) / data->k ) * .5f;
        if ( h > 1.f )  return RAYS_NORM( o0, p );
        if ( h < 0.f )  return RAYS_NORM( o1, p );
        point n0 = RAYS_NORM( o0, p ); point n1 = RAYS_NORM( o1, p );
        d0 = r_length_3( n0.x, n0.y, n0.z );
        d1 = r_length_3( n1.x, n1.y, n1.z );
        return Point( mix( d0 * n0.x, d1 * n1.x, h ), mix( d0 * n0.y, d1 * n1.y, h ), mix( d0 * n0.z, d1 * n1.z, h ) );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    glata_komunajo_2,
    {
        bazo_ptr o0 = obj + data->o[ 0 ]; bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        scalar h = ( 1.f + ( d0 - d1 ) / data->k ) * .5f;
        if ( h > 1.f )  return d0;
        if ( h < 0.f )  return d1;
        return mix( d0, d1, h ) + data->k * h * ( 1.f - h );
    },
    {
        bazo_ptr o0 = obj + data->o[ 0 ]; bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        scalar h = ( 1.f + ( d0 - d1 ) / data->k ) * .5f;
        if ( h > 1.f )  return RAYS_NORM( o0, p );
        if ( h < 0.f )  return RAYS_NORM( o1, p );
        point n0 = RAYS_NORM( o0, p ); point n1 = RAYS_NORM( o1, p );
        d0 = r_length_3( n0.x, n0.y, n0.z );
        d1 = r_length_3( n1.x, n1.y, n1.z );
        return Point( mix( d0 * n0.x, d1 * n1.x, h ), mix( d0 * n0.y, d1 * n1.y, h ), mix( d0 * n0.z, d1 * n1.z, h ) );
    } );


CREATE_OBJECT_TYPE_DEFINITION(
    movo,
    {
        bazo_ptr O = obj + data->o;
        point P;
        P.x = p.x - data->t.x;
        P.y = p.y - data->t.y;
        P.z = p.z - data->t.z;
        return RAYS_DIST( O, P );
    },
    {
        bazo_ptr O = obj + data->o;
        point P;
        P.x = p.x - data->t.x;
        P.y = p.y - data->t.y;
        P.z = p.z - data->t.z;
        return RAYS_NORM( O, P );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    rotacioX,
    {
        bazo_ptr O = obj + data->o;
        point P;
        P.y = data->cos_phi * p.y + data->sin_phi * p.z;
        P.z = -data->sin_phi * p.y + data->cos_phi * p.z;
        P.x = p.x;
        return RAYS_DIST( O, P );
    },
    {
        bazo_ptr O = obj + data->o;
        point P; point _P;
        P.y = data->cos_phi * p.y + data->sin_phi * p.z;
        P.z = -data->sin_phi * p.y + data->cos_phi * p.z;
        P.x = p.x;
        _P = RAYS_NORM( O, P );
        P.y = data->cos_phi * _P.y - data->sin_phi * _P.z;
        P.z = data->sin_phi * _P.y + data->cos_phi * _P.z;
        P.x = _P.x;
        return P;
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    rotacioY,
    {
        bazo_ptr O = obj + data->o;
        point P;
        P.z = data->cos_phi * p.z + data->sin_phi * p.x;
        P.x = -data->sin_phi * p.z + data->cos_phi * p.x;
        P.y = p.y;
        return RAYS_DIST( O, P );
    },
    {
        bazo_ptr O = obj + data->o;
        point P; point _P;
        P.z = data->cos_phi * p.z + data->sin_phi * p.x;
        P.x = -data->sin_phi * p.z + data->cos_phi * p.x;
        P.y = p.y;
        _P = RAYS_NORM( O, P );
        P.z = data->cos_phi * _P.z - data->sin_phi * _P.x;
        P.x = data->sin_phi * _P.z + data->cos_phi * _P.x;
        P.y = _P.y;
        return P;
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    rotacioZ,
    {
        bazo_ptr O = obj + data->o;
        point P;
        P.x = data->cos_phi * p.x + data->sin_phi * p.y;
        P.y = -data->sin_phi * p.x + data->cos_phi * p.y;
        P.z = p.z;
        return RAYS_DIST( O, P );
    },
    {
        bazo_ptr O = obj + data->o;
        point P; point _P;
        P.x = data->cos_phi * p.x + data->sin_phi * p.y;
        P.y = -data->sin_phi * p.x + data->cos_phi * p.y;
        P.z = p.z;
        _P = RAYS_NORM( O, P );
        P.x = data->cos_phi * _P.x - data->sin_phi * _P.y;
        P.y = data->sin_phi * _P.x + data->cos_phi * _P.y;
        P.z = _P.z;
        return P;
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    rotacioQ,
    {
        bazo_ptr O = obj + data->o;
        matrix Q; scalar temp;

        Q.x.x = data->q.x * data->q.x;
        Q.y.y = data->q.y * data->q.y;
        Q.z.z = data->q.z * data->q.z;
        temp = Q.x.x + Q.y.y + Q.z.z;
        Q.x.x -= temp;
        Q.y.y -= temp;
        Q.z.z -= temp;

        Q.x.y = data->q.x * data->q.y;
        temp = data->q.z * data->q_w;
        Q.y.x = Q.x.y + temp;
        Q.x.y -= temp;

        Q.y.z = data->q.y * data->q.z;
        temp = data->q.x * data->q_w;
        Q.z.y = Q.y.z + temp;
        Q.y.z -= temp;

        Q.z.x = data->q.z * data->q.x;
        temp = data->q.y * data->q_w;
        Q.x.z = Q.z.x + temp;
        Q.z.x -= temp;

        point P = p;
        P.x += 2.f * ( Q.x.x * p.x + Q.x.y * p.y + Q.x.z * p.z );
        P.y += 2.f * ( Q.y.x * p.x + Q.y.y * p.y + Q.y.z * p.z );
        P.z += 2.f * ( Q.z.x * p.x + Q.z.y * p.y + Q.z.z * p.z );
        return RAYS_DIST( O, P );
    },
    {
        bazo_ptr O = obj + data->o;
        matrix Q; scalar temp;

        Q.x.x = data->q.x * data->q.x;
        Q.y.y = data->q.y * data->q.y;
        Q.z.z = data->q.z * data->q.z;
        temp = Q.x.x + Q.y.y + Q.z.z;
        Q.x.x -= temp;
        Q.y.y -= temp;
        Q.z.z -= temp;

        Q.x.y = data->q.x * data->q.y;
        temp = data->q.z * data->q_w;
        Q.y.x = Q.x.y + temp;
        Q.x.y -= temp;

        Q.y.z = data->q.y * data->q.z;
        temp = data->q.x * data->q_w;
        Q.z.y = Q.y.z + temp;
        Q.y.z -= temp;

        Q.z.x = data->q.z * data->q.x;
        temp = data->q.y * data->q_w;
        Q.x.z = Q.z.x + temp;
        Q.z.x -= temp;

        point P = p;
        P.x += 2.f * ( Q.x.x * p.x + Q.x.y * p.y + Q.x.z * p.z );
        P.y += 2.f * ( Q.y.x * p.x + Q.y.y * p.y + Q.y.z * p.z );
        P.z += 2.f * ( Q.z.x * p.x + Q.z.y * p.y + Q.z.z * p.z );
        point N = RAYS_NORM( O, P );
        P = N;
        P.x += 2.f * ( Q.x.x * N.x + Q.y.x * N.y + Q.z.x * N.z );
        P.y += 2.f * ( Q.x.y * N.x + Q.y.y * N.y + Q.z.y * N.z );
        P.z += 2.f * ( Q.x.z * N.x + Q.y.z * N.y + Q.z.z * N.z );
        return P;
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    senfina_ripeto,
    {
        bazo_ptr o = obj + data->o;
        point a = data->a; scalar N = floorf( dot( a, p ) / dot( a, a ) + .5f );
        a.x = p.x - N * a.x;
        a.y = p.y - N * a.y;
        a.z = p.z - N * a.z;
        return RAYS_DIST( o, a );
    },
    {
        bazo_ptr o = obj + data->o;
        point a = data->a; scalar N = floorf( dot( a, p ) / dot( a, a ) + .5f );
        a.x = p.x - N * a.x;
        a.y = p.y - N * a.y;
        a.z = p.z - N * a.z;
        return RAYS_NORM( o, a );
    } );
};

namespace raymarching {

static size_t Width, Height;
static cudaSurfaceObject_t Surface_d;

static point *LightSource_d;
static primitives::bazo *Primitives_d;
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

static __device__ __inline__ point mul_point( const point &p, const scalar &s ) {
    return point{ s * p.x, s * p.y, s * p.z };
}

static __device__ __inline__ point add_point( const point &p1, const point &p2 ) {
    return point{ p1.x + p2.x, p1.y + p2.y, p1.z + p2.z };
}

static __device__ __inline__ scalar dot( const point &p1, const point &p2 ) {
    return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}

int Init( size_t width, size_t height, const cudaSurfaceObject_t &surface ) {
    Width = width;
    Height = height;
    Surface_d = surface;

    CUDA_ERROR( cudaMalloc( &LightSource_d, sizeof point ) );
    CUDA_ERROR( cudaMalloc( &Rays_d, Width * Height * sizeof ray ) );
    CUDA_ERROR( cudaMalloc( &Info_d, sizeof start_init_rays_info ) );
    return 1;
}

static __global__ void kernelInitPrimitives( primitives::bazo KERNEL_PTR Primitives, size_t PrimitivesNum ) {
    size_t x = RAYS_COORD_nD( x, 1 );

    if ( x < PrimitivesNum ) {
        primitives::bazo_ptr self = Primitives + x;
        CREATE_OBJECT_TYPE_PROCESSING_LISTING_2( self );
    }
}

int InitPrimitives( std::list< primitives::bazo_ptr > &Primitives, cudaStream_t stream ) {
    PrimitivesNum = Primitives.size();

    if ( Primitives_d ) CUDA_ERROR( cudaFree( Primitives_d ) );
    CUDA_ERROR( cudaMalloc( &Primitives_d, PrimitivesNum * sizeof primitives::bazo ) );

    size_t i = 0;
    for ( primitives::bazo_ptr ptr : Primitives ) {
        CUDA_ERROR( cudaMemcpyAsync( Primitives_d + i, ptr, sizeof primitives::bazo, cudaMemcpyHostToDevice, stream ) );
        ++i;
    }

    kernelInitPrimitives <<< grid( PrimitivesNum ), block_1d, 0, stream >>> ( Primitives_d, PrimitivesNum );

    return 1;
}

static __global__ void kernelLoad( start_init_rays_info KERNEL_PTR Info_d, ray KERNEL_PTR Rays ) {
    int64_t
        x = RAYS_COORD_nD( x, 2 ),
        y = RAYS_COORD_nD( y, 2 );

    __shared__ start_init_rays_info Info[ 1 ];
    if ( threadIdx.x == 0 && threadIdx.y == 0 )
        Info[ 0 ] = *Info_d;
    __syncthreads();

    if ( x < Info->Width && y < Info->Height ) {
        scalar
            X = .5f * ( 2 * x - int64_t( Info->Width ) + 1 ),
            Y = .5f * ( 2 * y - int64_t( Info->Height ) + 1 ),
            Z = Info->Depth;

        point pos;
        pos.x = X * Info->StartWVec.x + Y * Info->StartHVec.x;
        pos.y = X * Info->StartWVec.y + Y * Info->StartHVec.y;
        pos.z = X * Info->StartWVec.z + Y * Info->StartHVec.z;

        point delta_pos;
        delta_pos.x = Z * Info->StartDir.x;
        delta_pos.y = Z * Info->StartDir.y;
        delta_pos.z = Z * Info->StartDir.z;

        scalar R_1 = rnorm3df( pos.x + delta_pos.x, pos.y + delta_pos.y, pos.z + delta_pos.z );

        ray *self = Rays + y * Info->Width + x;
        self->d =
            //Info->StartDir;
            mul_point( add_point( pos, delta_pos ), R_1 );
        self->p = add_point( pos, Info->StartPos );
    }
}

int Load( point &LightSource, start_init_rays_info &Info, cudaStream_t stream ) {
    CUDA_ERROR( cudaMemcpyAsync( LightSource_d, &LightSource, sizeof point, cudaMemcpyHostToDevice, stream ) );

    CUDA_ERROR( cudaMemcpyAsync( Info_d, &Info, sizeof start_init_rays_info, cudaMemcpyHostToDevice, stream ) );

    kernelLoad <<< grid( Width, Height ), block_2d, 0, stream >>> ( Info_d, Rays_d );
    CUDA_ERROR( cudaStreamSynchronize( stream ) );

    return 1;
}

#define PRIMITIVES_PER_THREAD 2

static __global__ void kernelImageProcessing( cudaSurfaceObject_t image, size_t width, size_t height, size_t time, ray KERNEL_PTR Rays, point KERNEL_PTR LightSource, primitives::bazo KERNEL_PTR Primitives, size_t PrimitivesNum ) {
    size_t  x = RAYS_COORD_nD( x, 2 ),
            y = RAYS_COORD_nD( y, 2 ),
            id = PRIMITIVES_PER_THREAD * ( threadIdx.y * RAYS_BLOCK_2D_x + threadIdx.x );

    // RAYS_BLOCK_2D_x * RAYS_BLOCK_2D_y * PRIMITIVES_PER_THREAD >= PrimitivesNum
    __shared__ primitives::bazo curr_ptr[ RAYS_BLOCK_2D_x * RAYS_BLOCK_2D_y * PRIMITIVES_PER_THREAD ];
    if ( id < PrimitivesNum ) {
        primitives::bazo_ptr self = curr_ptr + id;

#pragma unroll
        for ( uint16_t i = 0; i < PRIMITIVES_PER_THREAD; ++i, ++self ) {
            *self = Primitives[ id + i ];
            //CREATE_OBJECT_TYPE_PROCESSING_LISTING_2( self );
        }
    }
    __syncthreads();

    if ( x < width && y < height ) {
        scalar curr_dist, ray_dist = 0;
        ray r = Rays[ y * width + x ];

        for ( size_t I = 0; true; ++I ) {
            curr_dist = RAYS_DIST( curr_ptr, r.p );

            r.p.x += curr_dist * r.d.x;
            r.p.y += curr_dist * r.d.y;
            r.p.z += curr_dist * r.d.z;

            if ( curr_dist < RAYS_MIN_DIST ) {
                point curr_norm, light = *LightSource;
                if ( curr_dist < 0.f ) {
                    curr_norm.x = -r.d.x;
                    curr_norm.y = -r.d.y;
                    curr_norm.z = -r.d.z;
                } else {
                    curr_norm = RAYS_NORM( curr_ptr, r.p );
                }

                if ( dot( curr_norm, r.d ) < 0.f ) {
                    scalar R_1 = r_length_3( curr_norm.x, curr_norm.y, curr_norm.z );

                    uint8_t LIGHT = 0xff * ( RAYS_MIN_LUM + .5f * ( RAYS_MAX_LUM - RAYS_MIN_LUM ) * ( 1.f + R_1 * dot( curr_norm, light ) ) );
                    uchar4 PIXEL = { LIGHT, LIGHT, LIGHT, 0xff };
                    surf2Dwrite( PIXEL, image, x * 4, y );
                    break;
                }
            }

            ray_dist += curr_dist;

            if ( ray_dist > RAYS_MAX_DIST || I >= RAYS_MAX_COUNTER ) {
                uchar3 COLOR = uchar3{ 0x00, 0x00, 0x00 };
                uchar4 PIXEL = RGB_PIXEL( COLOR );
                surf2Dwrite( PIXEL, image, x * 4, y );
                break;
            }
        }
    }
}

bool ImageProcessing( size_t time, cudaStream_t stream ) {
    kernelImageProcessing <<< grid( Width, Height ), block_2d, 0, stream >>>
        ( Surface_d, Width, Height, time, Rays_d, LightSource_d, Primitives_d, PrimitivesNum );
    CUDA_ERROR( cudaStreamSynchronize( stream ) );
    return true;
}

bool Quit() {
    cudaFree( Primitives_d );
    cudaFree( LightSource_d );
    cudaFree( Info_d );
    cudaFree( Rays_d );
    return true;
}

};