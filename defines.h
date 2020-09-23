#pragma once

// CUDA
#define CUDA_SET_GRID(c,n) ( c - 1 ) / Block##n##d.c + 1
#ifdef _DEBUG
#define CUDA_ERROR(__ERROR__) std::cout << (__ERROR__) << '\t'
#else
#define CUDA_ERROR(__ERROR__) __ERROR__
#endif // DEBUG


// PRIMITIVES
#define CREATE_OBJECT_TYPE_DESCRIPTION(__TYPE__,__STRUCT__)                     \
class __TYPE__ {                                                                \
protected:                                                                      \
    typedef __STRUCT__ data_struct;                                             \
    template< typename ARG0, typename... ARGN >                                 \
    static __host__ base_ptr emplace( byte *data, ARG0 arg, ARGN... args ) {    \
        memcpy( data, &arg, sizeof ARG0 );                                      \
        return emplace( data + sizeof ARG0, args... );                          \
    }                                                                           \
    template< typename ARG0 >                                                   \
    static __host__ base_ptr emplace( byte *data, ARG0 arg ) {                  \
        memcpy( data, &arg, sizeof ARG0 );                                      \
        return nullptr;                                                         \
    }                                                                           \
public:                                                                         \
    static __host__ base_ptr create( data_struct& );                            \
    template< typename... ARGN >                                                \
    static __host__ base_ptr create_from( ARGN... args ) {                      \
        data_struct NEW;                                                        \
        emplace( reinterpret_cast< byte* >( &NEW ), args... );                  \
        return create( NEW );                                                   \
    }                                                                           \
    static __device__ scalar dist( byte*, const point& p );                     \
    static __device__ point norm( byte*, const point& p );                      \
};

#define CREATE_OBJECT_TYPE_DEFINITION(__TYPE__,__DIST__,__NORM__)               \
__host__ base_ptr __TYPE__##::create( __TYPE__##::data_struct &data ) {         \
    base_ptr NEW = new base( type_##__TYPE__ );                                 \
    memcpy( NEW->data, &data, sizeof data_struct );                             \
    return NEW;                                                                 \
}                                                                               \
__device__ scalar __TYPE__##::dist( byte *_data, const point &p ) {             \
    data_struct *data = reinterpret_cast< data_struct* >( _data );              \
    __DIST__                                                                    \
}                                                                               \
__device__ point __TYPE__##::norm( byte *_data, const point &p ) {              \
    data_struct *data = reinterpret_cast< data_struct* >( _data );              \
    __NORM__                                                                    \
}

#define CREATE_OBJECT_TYPE_PROCESSING(type,func)                                \
case primitives::type_##type:                                                   \
    curr_##func = primitives::##type##::##func##( curr_object.data, r.p );            \
    break;

// RAYMARCHING
#define RAYS_BLOCK_1D_x 256

#define RAYS_BLOCK_2D_x 16
#define RAYS_BLOCK_2D_y 16

#define RAYS_BLOCK_3D_x 8
#define RAYS_BLOCK_3D_y 8
#define RAYS_BLOCK_3D_z 4

#define RAYS_COORD_nD(c,n) blockIdx.##c * RAYS_BLOCK_##n##D_##c + threadIdx.##c

#define RAYS_MAX_DIST 10000.f
#define RAYS_MIN_DIST .1f

#define RAYS_MAX_LUM .9f
#define RAYS_MIN_LUM .0f

#define KERNEL_PTR *__restrict__

#define RGB_PIXEL(p) (uchar4{ (p.x), (p.y), (p.z), 0xff });