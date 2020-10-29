#include "cuda_rays.cuh"
#include <cuda_device_runtime_api.h>

#define _KERNEL             __global__ void
#define _PTR                *__restrict__

#define _CUDA(__ERROR__)    {_last_cuda_error = cudaError_t(__ERROR__); CUDA_CHECK(_last_cuda_error);}
#define _RETURN             return _last_cuda_error;
#define _CAST(__TYPE__)     reinterpret_cast<__TYPE__>

#define _PRIM               _CAST(primitives::bazo_ptr)(Primitives_d)
#define _RAYS               _CAST(ray*)(Rays_d)
#define _INFO               _CAST(rays_info*)(Info_d)

#define grid_1d(X)          (dim3( ( (X) - 1 ) / RAYS_BLOCK_1D_x + 1 ))
#define grid_2d(X,Y)        (dim3( ( (X) - 1 ) / RAYS_BLOCK_2D_x + 1, ( (Y) - 1 ) / RAYS_BLOCK_2D_y + 1 ))

#define block_1d            (dim3((RAYS_BLOCK_1D_x)))
#define block_2d            (dim3((RAYS_BLOCK_2D_x),(RAYS_BLOCK_2D_y)))

namespace cuda {


// Kernels definitions
namespace kernel {


static __device__ __inline__ point new_point( const scalar &x, const scalar &y, const scalar &z ) {
    return point{ x, y, z };
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

static __device__ __inline__ scalar mix( const scalar &a, const scalar &b, const scalar &x ) {
    return a + x * ( b - a );
}

static _KERNEL Process( size_t Width, size_t Height, rays_info _PTR Info_d, ray _PTR Rays,
                        primitives::bazo _PTR Primitives_d, size_t PrimitivesNum, cudaSurfaceObject_t Image ) {
    size_t  x = CUDA_RAYS_COORD_nD( x, 2 ),
            y = CUDA_RAYS_COORD_nD( y, 2 ),
            id = RAYS_PRIMITIVES_PER_THREAD * ( threadIdx.y * RAYS_BLOCK_2D_x + threadIdx.x );

    // RAYS_BLOCK_2D_x * RAYS_BLOCK_2D_y * PRIMITIVES_PER_THREAD >= PrimitivesNum
    __shared__ primitives::bazo curr_ptr[ RAYS_BLOCK_2D_x * RAYS_BLOCK_2D_y * RAYS_PRIMITIVES_PER_THREAD ];
    if ( id < PrimitivesNum ) {
        primitives::bazo_ptr self = curr_ptr + id;

#pragma unroll
        for ( uint16_t i = 0; i < RAYS_PRIMITIVES_PER_THREAD; ++i, ++self ) {
            *self = Primitives_d[ id + i ];
            //CREATE_OBJECT_TYPE_PROCESSING_LISTING_2( self );
        }
    }
    __syncthreads();

    if ( x < Width && y < Height ) {
        scalar curr_dist, ray_dist = 0;
        ray r = Rays[ y * Width + x ];
        uchar4 PIXEL = { 0x00, 0x00, 0x00, 0xff };
        point curr_norm, light =  Info_d->LightSource; //point{ 1.f, 0.f, 0.f };

#pragma unroll
        for ( size_t I = 0; I < RAYS_MAX_COUNTER; ++I ) {
            curr_dist = RAYS_DIST( curr_ptr, r.p );

            if ( curr_dist < RAYS_MIN_DIST ) {
                if ( curr_dist < 0.f ) {
                    curr_norm.x = -r.d.x;
                    curr_norm.y = -r.d.y;
                    curr_norm.z = -r.d.z;
                } else {
                    curr_norm = RAYS_NORM( curr_ptr, r.p );
                }

                if ( dot( curr_norm, r.d ) < 0.f ) {
                    scalar R_1 = r_length_3( curr_norm.x, curr_norm.y, curr_norm.z ), N_L;
                    curr_norm = mul_point( curr_norm, R_1 );
                    N_L = dot( curr_norm, light );

                    float
                        AMBIENT = 1.f,
                        SHADOW = 1.f,
                        OCCLUSION = 0.f,
                        SCA = 1.f;
                    point p = r.p;

                    ray_dist = RAYS_MIN_DIST;

#pragma unroll
                    for ( size_t J = 0; J < 5; ++J ) {
                        curr_dist = RAYS_DIST( curr_ptr, p );
                        OCCLUSION += ( ray_dist - curr_dist ) * SCA;
                        SCA *= .95;

                        p.x += .04 * curr_dist * curr_norm.x;
                        p.y += .04 * curr_dist * curr_norm.y;
                        p.z += .04 * curr_dist * curr_norm.z;
                        ray_dist += curr_dist;
                    }

                    p = r.p;
                    p.x += RAYS_MIN_DIST * light.x;
                    p.y += RAYS_MIN_DIST * light.y;
                    p.z += RAYS_MIN_DIST * light.z;
                    ray_dist = RAYS_MIN_DIST;

#pragma unroll
                    for ( size_t J = 0; J < RAYS_MAX_COUNTER; ++J ) {
                        curr_dist = RAYS_DIST( curr_ptr, p );

                        if ( 10 * curr_dist < RAYS_MIN_DIST ) {
                            // NO LIGHT
                            SHADOW = 0x00;
                            break;
                        }

                        SHADOW = min( SHADOW, RAYS_SHADOW * curr_dist / ray_dist );

                        p.x += curr_dist * light.x;
                        p.y += curr_dist * light.y;
                        p.z += curr_dist * light.z;
                        ray_dist += curr_dist;

                        if ( ray_dist < RAYS_MAX_DIST ) {
                            // LIGHT
                            break;
                        }
                    }

                    float3 MATERIAL = { 1.f, 1.f, 1.f };
                    uint8_t LIGHT =
                        //0xff * ( RAYS_MIN_LUM * () + .5f * ( RAYS_MAX_LUM - RAYS_MIN_LUM ) * ( 1.f + R_1 * N_L ) );
                        0xff * ( RAYS_MIN_LUM * AMBIENT * OCCLUSION + ( RAYS_MAX_LUM - RAYS_MIN_LUM ) * ( N_L > 0.f ? N_L : 0.f ) * SHADOW );
                    PIXEL = {
                         raw_byte( LIGHT * MATERIAL.x ),
                         raw_byte( LIGHT * MATERIAL.y ),
                         raw_byte( LIGHT * MATERIAL.z ),
                         0xff
                    };
                    break;
                }
            }

            r.p.x += curr_dist * r.d.x;
            r.p.y += curr_dist * r.d.y;
            r.p.z += curr_dist * r.d.z;
            ray_dist += curr_dist;

            if ( ray_dist >= RAYS_MAX_DIST ) {
                break;
            }
        }

        surf2Dwrite( PIXEL, Image, x * 4, y );
    }
}

static _KERNEL SetPrimitives( primitives::bazo _PTR Primitives, size_t PrimitivesNum ) {
    size_t x = CUDA_RAYS_COORD_nD( x, 1 );
    
    if ( x < PrimitivesNum ) {
        primitives::bazo_ptr self = Primitives + x;
        CREATE_OBJECT_TYPE_PROCESSING_LISTING_2( self );
    }
}

static _KERNEL SetRays( size_t Width, size_t Height, rays_info _PTR Info_d, ray _PTR Rays_d ) {
    int64_t
        x = CUDA_RAYS_COORD_nD( x, 2 ),
        y = CUDA_RAYS_COORD_nD( y, 2 );

    __shared__ rays_info Info[ 1 ];
    if ( threadIdx.x == 0 && threadIdx.y == 0 )
        Info[ 0 ] = *Info_d;
    __syncthreads();

    if ( x < Width && y < Height ) {
        scalar
            X = .5f * float( 2 * x - int64_t( Width ) + 1 ),
            Y = .5f * float( 2 * y - int64_t( Height ) + 1 ),
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

        ray *self = Rays_d + y * Width + x;
        self->d = mul_point( add_point( pos, delta_pos ), R_1 );
        self->p = add_point( pos, Info->StartPos );
    }
}


};


// TYPE_LIST
CREATE_OBJECT_TYPE_DEFINITION(
    portanta_sfero,
    {
        point P;
        P.x = p.x - data->t.x;
        P.y = p.y - data->t.y;
        P.z = p.z - data->t.z;
        scalar d = length_3( P.x, P.y, P.z ) - data->r;
        primitives::bazo_ptr o = obj + data->o;
        if ( d <= RAYS_MIN_DIST )   return RAYS_DIST( o, P );
        else                        return d;
    },
    {
        point P;
        P.x = p.x - data->t.x;
        P.y = p.y - data->t.y;
        P.z = p.z - data->t.z;
        primitives::bazo_ptr o = obj + data->o;
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
        return q.x > q.z ? ( q.x > q.y ? kernel::new_point( p.x > 0.f ? 1.f : -1.f, 0.f, 0.f ) : kernel::new_point( 0.f, p.y > 0.f ? 1.f : -1.f, 0.f ) ) : ( q.y > q.z ? kernel::new_point( 0.f, p.y > 0.f ? 1.f : -1.f, 0.f ) : kernel::new_point( 0.f, 0.f, p.z > 0.f ? 1.f : -1.f ) );
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
        
        return q.x > q.y ? kernel::new_point( p.x, p.y, 0.f ) : kernel::new_point( 0.f, 0.f, p.z > 0.f ? 1.f : -1.f );
    } );

CREATE_OBJECT_TYPE_DEFINITION(
    ebeno,
    {
        return kernel::dot( data->n, p );
    },
    {
        return data->n;
    } );

CREATE_OBJECT_TYPE_DEFINITION(
    kunigajo_2,
    {
        primitives::bazo_ptr o0 = obj + data->o[ 0 ]; primitives::bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        return min( d0, d1 );
    },
    {
        primitives::bazo_ptr o0 = obj + data->o[ 0 ]; primitives::bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );

        if ( d0 < d1 )  return RAYS_NORM( o0, p );
        else            return RAYS_NORM( o1, p );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    kunigajo_3,
    {
        primitives::bazo_ptr o = obj + data->o[ 0 ];
        scalar d = RAYS_DIST( o, p );

        o = obj + data->o[ 1 ];
        d = min( d, RAYS_DIST( o, p ) );

        o = obj + data->o[ 2 ];
        return min( d, RAYS_DIST( o, p ) );
    },
    {
        counter i_min = 0;
        primitives::bazo_ptr o = obj + data->o[ 0 ];
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
        primitives::bazo_ptr o0 = obj + data->o[ 0 ]; primitives::bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        return min( d0, d1 );
    },
    {
        primitives::bazo_ptr o0 = obj + data->o[ 0 ]; primitives::bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );

        if ( d0 < d1 )  return RAYS_NORM( o0, p );
        else            return RAYS_NORM( o1, p );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    komunajo_2,
    {
        primitives::bazo_ptr o0 = obj + data->o[ 0 ]; primitives::bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        return max( d0, d1 );
    },
    {
        primitives::bazo_ptr o0 = obj + data->o[ 0 ]; primitives::bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );

        if ( d0 > d1 )  return RAYS_NORM( o0, p );
        else            return RAYS_NORM( o1, p );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    komunajo_3,
    {
        primitives::bazo_ptr o = obj + data->o[ 0 ];
        scalar d = RAYS_DIST( o, p );

        o = obj + data->o[ 1 ];
        d = max( d, RAYS_DIST( o, p ) );

        o = obj + data->o[ 2 ];
        return max( d, RAYS_DIST( o, p ) );
    },
    {
        counter i_max = 0;
        primitives::bazo_ptr o = obj + data->o[ 0 ];
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
        primitives::bazo_ptr o0 = obj + data->o[ 0 ]; primitives::bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        return max( d0, d1 );
    },
    {
        primitives::bazo_ptr o0 = obj + data->o[ 0 ]; primitives::bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );

        if ( d0 > d1 )  return RAYS_NORM( o0, p );
        else            return RAYS_NORM( o1, p );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    komplemento,
    {
        primitives::bazo_ptr O = obj + data->o;
        scalar D = RAYS_DIST( O, p );
        return -D;
    },
    {
        primitives::bazo_ptr O = obj + data->o;
        point N = RAYS_NORM( O, p );
        return kernel::new_point( -N.x, -N.y, -N.z );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    glata_kunigajo_2,
    {
        primitives::bazo_ptr o0 = obj + data->o[ 0 ]; primitives::bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        scalar h = ( 1.f - ( d0 - d1 ) / data->k ) * .5f;
        if ( h > 1.f )  return d0;
        if ( h < 0.f )  return d1;
        return kernel::mix( d0, d1, h ) - data->k * h * ( 1.f - h );
    },
    {
        primitives::bazo_ptr o0 = obj + data->o[ 0 ]; primitives::bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        scalar h = ( 1.f - ( d0 - d1 ) / data->k ) * .5f;
        if ( h > 1.f )  return RAYS_NORM( o0, p );
        if ( h < 0.f )  return RAYS_NORM( o1, p );
        point n0 = RAYS_NORM( o0, p ); point n1 = RAYS_NORM( o1, p );
        d0 = r_length_3( n0.x, n0.y, n0.z );
        d1 = r_length_3( n1.x, n1.y, n1.z );
        return kernel::new_point( kernel::mix( d0 * n0.x, d1 * n1.x, h ), kernel::mix( d0 * n0.y, d1 * n1.y, h ), kernel::mix( d0 * n0.z, d1 * n1.z, h ) );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    glata_komunajo_2,
    {
        primitives::bazo_ptr o0 = obj + data->o[ 0 ]; primitives::bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        scalar h = ( 1.f + ( d0 - d1 ) / data->k ) * .5f;
        if ( h > 1.f )  return d0;
        if ( h < 0.f )  return d1;
        return kernel::mix( d0, d1, h ) + data->k * h * ( 1.f - h );
    },
    {
        primitives::bazo_ptr o0 = obj + data->o[ 0 ]; primitives::bazo_ptr o1 = obj + data->o[ 1 ];
        scalar d0 = RAYS_DIST( o0, p ); scalar d1 = RAYS_DIST( o1, p );
        scalar h = ( 1.f + ( d0 - d1 ) / data->k ) * .5f;
        if ( h > 1.f )  return RAYS_NORM( o0, p );
        if ( h < 0.f )  return RAYS_NORM( o1, p );
        point n0 = RAYS_NORM( o0, p ); point n1 = RAYS_NORM( o1, p );
        d0 = r_length_3( n0.x, n0.y, n0.z );
        d1 = r_length_3( n1.x, n1.y, n1.z );
        return kernel::new_point( kernel::mix( d0 * n0.x, d1 * n1.x, h ), kernel::mix( d0 * n0.y, d1 * n1.y, h ), kernel::mix( d0 * n0.z, d1 * n1.z, h ) );
    } );


CREATE_OBJECT_TYPE_DEFINITION(
    movo,
    {
        primitives::bazo_ptr O = obj + data->o;
        point P;
        P.x = p.x - data->t.x;
        P.y = p.y - data->t.y;
        P.z = p.z - data->t.z;
        return RAYS_DIST( O, P );
    },
    {
        primitives::bazo_ptr O = obj + data->o;
        point P;
        P.x = p.x - data->t.x;
        P.y = p.y - data->t.y;
        P.z = p.z - data->t.z;
        return RAYS_NORM( O, P );
    } );
CREATE_OBJECT_TYPE_DEFINITION(
    rotacioX,
    {
        primitives::bazo_ptr O = obj + data->o;
        point P;
        P.y = data->cos_phi * p.y + data->sin_phi * p.z;
        P.z = -data->sin_phi * p.y + data->cos_phi * p.z;
        P.x = p.x;
        return RAYS_DIST( O, P );
    },
    {
        primitives::bazo_ptr O = obj + data->o;
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
        primitives::bazo_ptr O = obj + data->o;
        point P;
        P.z = data->cos_phi * p.z + data->sin_phi * p.x;
        P.x = -data->sin_phi * p.z + data->cos_phi * p.x;
        P.y = p.y;
        return RAYS_DIST( O, P );
    },
    {
        primitives::bazo_ptr O = obj + data->o;
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
        primitives::bazo_ptr O = obj + data->o;
        point P;
        P.x = data->cos_phi * p.x + data->sin_phi * p.y;
        P.y = -data->sin_phi * p.x + data->cos_phi * p.y;
        P.z = p.z;
        return RAYS_DIST( O, P );
    },
    {
        primitives::bazo_ptr O = obj + data->o;
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
        primitives::bazo_ptr O = obj + data->o;
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
        primitives::bazo_ptr O = obj + data->o;
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
        primitives::bazo_ptr o = obj + data->o;
        point a = data->a; scalar N = floorf( kernel::dot( a, p ) / kernel::dot( a, a ) + .5f );
        a.x = p.x - N * a.x;
        a.y = p.y - N * a.y;
        a.z = p.z - N * a.z;
        return RAYS_DIST( o, a );
    },
    {
        primitives::bazo_ptr o = obj + data->o;
        point a = data->a; scalar N = floorf( kernel::dot( a, p ) / kernel::dot( a, a ) + .5f );
        a.x = p.x - N * a.x;
        a.y = p.y - N * a.y;
        a.z = p.z - N * a.z;
        return RAYS_NORM( o, a );
    } );


int raymarching::Init( rays_Init_args ) {
    for ( size_t n = 0; n < CUDA_RAYS_STREAM_NUM; ++n ) {
        _CUDA( cudaStreamCreate( _stream + n ) )
    }
    _default_stream = _stream[ CUDA_RAYS_DEFAULT_STREAM ];
    
    for ( size_t n = 0; n < CUDA_RAYS_EVENT_NUM; ++n ) {
        _CUDA( cudaEventCreate( _event + n ) )
    }
    
    Width = width;
    Height = height;

    _CUDA( cudaMalloc( &Rays_d, Width * Height * sizeof ray ) )
    _CUDA( cudaMalloc( &Info_d, sizeof rays_info ) )
    Primitives_d = nullptr;
    
    _RETURN
}

int raymarching::Process( rays_Process_args ) {
    _CUDA( cudaEventRecord( _event[ 0 ], _default_stream ) )
    
    kernel::Process <<< grid_2d( Width, Height ), block_2d, 0, _default_stream >>>
        ( Width, Height, _INFO, _RAYS, _PRIM, PrimitivesNum, _surface );
    
    _CUDA( cudaEventRecord( _event[ 1 ], _default_stream ) )
    _CUDA( cudaStreamSynchronize( _default_stream ) )
    _CUDA( cudaEventElapsedTime( &_last_process_time, _event[ 0 ], _event[ 1 ] ) )
    
    return uint32_t( _last_process_time );
}

int raymarching::Quit( rays_Quit_args ) {
    _CUDA( cudaDestroySurfaceObject( _surface ) )
    _CUDA( cudaGraphicsUnmapResources( 1, &_resource, _default_stream ) )
    
    _CUDA( cudaFree( Primitives_d ) )
    _CUDA( cudaFree( Rays_d ) )
    _CUDA( cudaFree( Info_d ) )
    
    _RETURN
}


int raymarching::SetInfo( rays_SetInfo_args ) {
    _CUDA( cudaMemcpyAsync( Info_d, &info, sizeof rays_info, cudaMemcpyHostToDevice, _default_stream ) )
    _CUDA( SetRays() )
    _CUDA( cudaStreamSynchronize( _default_stream ) )

    
    _RETURN
}

int raymarching::SetTexture( rays_SetTexture_args ) {
    _CUDA( cudaGraphicsGLRegisterImage( &_resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore ) )
    _CUDA( cudaGraphicsMapResources( 1, &_resource, _default_stream ) )
    
    _resource_desc.resType = cudaResourceTypeArray;
    _CUDA( cudaGraphicsSubResourceGetMappedArray( &_resource_desc.res.array.array, _resource, 0, 0 ) )
    
    _CUDA( cudaCreateSurfaceObject( &_surface, &_resource_desc ) )
    
    _RETURN
}

int raymarching::SetPrimitives( rays_SetPrimitives_args ) {
    PrimitivesNum = Primitives_h.size();
    
    if ( Primitives_d ) {
        _CUDA( cudaFree( Primitives_d ) )
    }
    
    _CUDA( cudaMalloc( &Primitives_d, PrimitivesNum * sizeof primitives::bazo ) )
    _CUDA( cudaMemcpyAsync( Primitives_d, Primitives_h.data(), PrimitivesNum * sizeof primitives::bazo, cudaMemcpyHostToDevice, _default_stream ) )
    
    kernel::SetPrimitives <<< grid_1d( PrimitivesNum ), block_1d, 0, _default_stream >>>
        ( _PRIM, PrimitivesNum );
    _CUDA( cudaStreamSynchronize( _default_stream ) )
    
    _RETURN
}

int raymarching::SetRays( rays_SetRays_args ) {
    kernel::SetRays <<< grid_2d( Width, Height ), block_2d, 0, _default_stream >>>
        ( Width, Height, _INFO, _RAYS );
    
    _RETURN
}

};