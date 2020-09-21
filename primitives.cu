#include "rays.h"
#include <cuda_device_runtime_api.h>

#define DATA (reinterpret_cast<data_struct*>(data))

namespace primitives {

static __device__ __inline__ point mul_point( const point& p, const scalar& s ) {
    return point{ s * p.x, s * p.y, s * p.z };
}

template< typename __TYPE__ >
union byte_cast {
    __TYPE__ data;
    byte source[ sizeof __TYPE__ ];
};

// SPHERE

sphere::sphere( const point& center, const scalar& radius ) {
    byte_cast< data_struct > _data;
    _data.data.r = radius;
    _data.data.c = center;

    for ( size_t i = 0; i < sizeof data_struct; ++i )
        data[ i ] = _data.source[ i ];
}


__device__ scalar sphere::dist( const point& p ) {
    point c = DATA->c;
    return norm3df( p.x - c.x, p.y - c.y, p.z - c.z ) - DATA->r;
}

__device__ point sphere::norm( const point& p ) {
    point c = DATA->c, dp = point{ p.x - c.x, p.y - c.y , p.z - c.z };
    return mul_point( dp, rnorm3df( dp.x, dp.y, dp.z ) );
}

};