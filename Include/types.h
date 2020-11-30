#pragma once

typedef unsigned char                   raw_byte;
typedef long long                       coord;
typedef long                            counter;
typedef float                           scalar;

typedef struct { scalar x, y, z; }      point;
typedef struct { point x, y, z; }       matrix;

typedef struct { scalar x, y, z, w; }   point4;
typedef struct { point4 x, y, z, w; }   matrix4;

typedef struct { point p, d; }          ray;
typedef struct {
    size_t Depth;
    point  LightSource;
    point  StartPos, StartDir, StartWVec, StartHVec;
}                                       rays_info;

namespace primitives {

struct bazo;
typedef bazo *bazo_ptr;

typedef scalar( *dist_func )( bazo_ptr, const point & );
typedef point ( *norm_func )( bazo_ptr, const point & );

enum object_type {
    type_nenio =                        0x0000,
    type_portanta_sfero,                // BROKEN ILLUMINATION
    type_sfero,
    type_kubo,
    type_cilindro,
    
    type_ebeno =                        0x0080,
    
    type_kunigajo_2 =                   0x0100,
    type_kunigajo_3,
    type_kunigajo_4,                    // YET NOT IMPLEMENTED
    type_komunajo_2,
    type_komunajo_3,
    type_komunajo_4,                    // YET NOT IMPLEMENTED
    type_komplemento,
    type_glata_kunigajo_2,
    type_glata_komunajo_2,
    
    type_movo =                         0x0200,
    type_rotacioX,
    type_rotacioY,
    type_rotacioZ,
    type_rotacioQ,
    type_senfina_ripeto
};

struct bazo {
    enum object_type                    type;
    raw_byte                            data[PRIMITIVE_PAYLOAD];
    dist_func                           dist;
    norm_func                           norm;
};

#ifndef __CUDACC_RTC__
template <typename __STRUCT__>
bazo create( enum object_type _type, __STRUCT__ *_arg ) {
    primitives::bazo NEW;
    NEW.type = _type;
    NEW.dist = nullptr;
    NEW.norm = nullptr;
    memcpy( NEW.data, _arg, sizeof( __STRUCT__ ) );
    return NEW;
}
#endif

};