#pragma once

#define rays_Init_args          const size_t &width, const size_t &height
#define rays_Process_args
#define rays_Quit_args
#define rays_SetInfo_args       const rays_info &info
#define rays_SetTexture_args    GLuint texture
#define rays_SetPrimitives_args
#define rays_SetRays_args

// PRIMITIVES
#define CREATE_OBJECT_TYPE_DESCRIPTION(__TYPE__,__STRUCT__)                                         \
class __TYPE__ {                                                                                    \
protected:                                                                                          \
    typedef __STRUCT__ data_struct;                                                                 \
    template< typename ARG0, typename... ARGN >                                                     \
    static __host__ ::primitives::bazo_ptr emplace( raw_byte *data, ARG0 arg, ARGN... args ) {          \
        memcpy( data, &arg, sizeof ARG0 );                                                          \
        return emplace( data + sizeof ARG0, args... );                                              \
    }                                                                                               \
    template< typename ARG0 >                                                                       \
    static __host__ primitives::bazo_ptr emplace( raw_byte *data, ARG0 arg ) {                          \
        memcpy( data, &arg, sizeof ARG0 );                                                          \
        return nullptr;                                                                             \
    }                                                                                               \
public:                                                                                             \
    template< typename... ARGN >                                                                    \
    static __host__ primitives::bazo create_from( ARGN... args ) {                                  \
        data_struct NEW;                                                                            \
        memset( &NEW, 0x00, sizeof data_struct );                                                   \
        emplace( reinterpret_cast< raw_byte* >( &NEW ), args... );                                      \
        return create( NEW );                                                                       \
    }                                                                                               \
    static __host__ primitives::bazo __TYPE__##::create( data_struct &data ) {                      \
        primitives::bazo NEW( primitives::type_##__TYPE__ );                                        \
        memcpy( NEW.data, &data, sizeof data_struct );                                              \
        return NEW;                                                                                 \
    }                                                                                               \
    static __device__ __forceinline__ scalar dist( primitives::bazo_ptr, const point& p );          \
    static __device__ __forceinline__ point norm( primitives::bazo_ptr, const point& p );           \
};

#define CREATE_OBJECT_TYPE_DEFINITION(__TYPE__,__DIST__,__NORM__)                                   \
__device__ __forceinline__ scalar                                                                   \
__TYPE__##::dist( primitives::bazo_ptr obj, const point &p ) {                                      \
    data_struct *data = reinterpret_cast<data_struct*>( obj->data );                                \
    __DIST__                                                                                        \
}                                                                                                   \
__device__ __forceinline__ point                                                                    \
__TYPE__##::norm( primitives::bazo_ptr obj, const point &p ) {                                      \
    data_struct *data = reinterpret_cast<data_struct*>( obj->data );                                \
    __NORM__                                                                                        \
}

#define CREATE_OBJECT_TYPE_PROCESSING_2(__SELF__,__TYPE__)                                          \
case primitives::type_##__TYPE__:                                                                   \
    __SELF__->dist = ##__TYPE__##::dist;                                                            \
    __SELF__->norm = ##__TYPE__##::norm;                                                            \
    break;

#define CREATE_OBJECT_TYPE_PROCESSING_LISTING_2(__SELF__)                                           \
switch ( __SELF__->type ) {                                                                         \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, portanta_sfero );                                    \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, sfero );                                             \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, kubo );                                              \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, cilindro );                                          \
                                                                                                    \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, ebeno );                                             \
                                                                                                    \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, kunigajo_2 );                                        \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, kunigajo_3 );                                        \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, kunigajo_4 );                                        \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, komunajo_2 );                                        \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, komunajo_3 );                                        \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, komunajo_4 );                                        \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, komplemento );                                       \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, glata_kunigajo_2 );                                  \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, glata_komunajo_2 );                                  \
                                                                                                    \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, movo );                                              \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, rotacioX );                                          \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, rotacioY );                                          \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, rotacioZ );                                          \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, rotacioQ );                                          \
    CREATE_OBJECT_TYPE_PROCESSING_2( __SELF__, senfina_ripeto );                                    \
    default: __SELF__->dist = nullptr; __SELF__->norm = nullptr;                                     \
}

#define RAYS_DIST(__SELF__,__POINT__) ((__SELF__)->dist((__SELF__),(__POINT__)))
#define RAYS_NORM(__SELF__,__POINT__) ((__SELF__)->norm((__SELF__),(__POINT__)))


// RAYMARCHING
#define RAYS_PRIMITIVES_PER_THREAD 2

#define RAYS_BLOCK_1D_x     128

#define RAYS_BLOCK_2D_x     16
#define RAYS_BLOCK_2D_y     8

#define RAYS_MAX_COUNTER    1000

#define RAYS_MAX_DIST       10000.f
#define RAYS_MIN_DIST       .01f

#define RAYS_MAX_LUM        .9f
#define RAYS_MIN_LUM        .1f

#define RAYS_SHADOW         64.f