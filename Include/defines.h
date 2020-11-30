#pragma once

#include "config.h"

#define STRINGIFY(__STR__)      #__STR__

#define rays_Init_args          const size_t &width, const size_t &height
#define rays_Process_args
#define rays_Quit_args
#define rays_SetInfo_args       const rays_info &info
#define rays_UnsetInfo_args
#define rays_SetTexture_args    unsigned int texture
#define rays_SetPrimitives_args
#define rays_SetRays_args

// PRIMITIVES
#ifndef CREATE_OBJECT_TYPE_DESCRIPTION
#define CREATE_OBJECT_TYPE_DESCRIPTION(__TYPE__,__STRUCT__)                                         \
class __TYPE__ {                                                                                    \
protected:                                                                                          \
public:                                                                                             \
    typedef __STRUCT__ data_struct;                                                                 \
    static inline bazo create_from( data_struct data ) {                                            \
        return primitives::create( primitives::type_##__TYPE__, &data );                            \
    }                                                                                               \
};
#endif

#define CREATE_OBJECT_TYPE_DEFINITION(__TYPE__,__DIST__,__NORM__)                                   \
__device__ __forceinline__ scalar                                                                   \
__TYPE__##_dist( primitives::bazo_ptr obj, const point &p ) {                                       \
    __TYPE__    ::data_struct *data = reinterpret_cast<__TYPE__  ::data_struct*>( obj->data );      \
    __DIST__                                                                                        \
}                                                                                                   \
__device__ __forceinline__ point                                                                    \
__TYPE__##_norm( primitives::bazo_ptr obj, const point &p ) {                                       \
    __TYPE__    ::data_struct *data = reinterpret_cast<__TYPE__  ::data_struct*>( obj->data );      \
    __NORM__                                                                                        \
}

#define CREATE_OBJECT_TYPE_PROCESSING_2(__SELF__,__TYPE__)                                          \
case primitives::type_##__TYPE__:                                                                   \
    __SELF__->dist = primitives::   __TYPE__##_dist;                                                \
    __SELF__->norm = primitives::   __TYPE__##_norm;                                                \
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
    default: /* do nothing */                                                                       \
}

#define RAYS_DIST(__SELF__,__POINT__) ((__SELF__)->dist((__SELF__),(__POINT__)))
#define RAYS_NORM(__SELF__,__POINT__) ((__SELF__)->norm((__SELF__),(__POINT__)))


// RAYMARCHING
#define PRIMITIVE_PAYLOAD           24

#define RAYS_PRIMITIVES_PER_THREAD  2

#define RAYS_BLOCK_1D_x             128

#define RAYS_BLOCK_2D_x             16
#define RAYS_BLOCK_2D_y             8

#define RAYS_MAX_COUNTER            1000

#define RAYS_MAX_DIST               10000.f
#define RAYS_MIN_DIST               .01f

#define RAYS_MAX_LUM                .9f
#define RAYS_MIN_LUM                .1f

#define RAYS_SHADOW                 64.f