#pragma once

#include "defines.h"

#include <vector>
#include <string>
#include <iostream>

#include <SDL.h>
#include <SDL_opengl.h>

typedef unsigned char raw_byte;

typedef long counter;

typedef float scalar;

typedef struct { scalar x, y, z; } point;
typedef struct { point  x, y, z; } matrix;

typedef struct { scalar x, y, z, w; } point4;
typedef struct { point4 x, y, z, w; } matrix4;

typedef struct { point  p, d; } ray;
typedef struct {
    size_t  Depth;
    point   LightSource;
    point   StartPos, StartDir, StartWVec, StartHVec;
} rays_info;

namespace primitives {

struct bazo;
typedef bazo* bazo_ptr;

typedef scalar( *dist_func )( bazo_ptr, const point& );
typedef point( *norm_func )( bazo_ptr, const point& );

enum object_type {
    type_nenio = 0x0000,
    type_portanta_sfero,        // BROKEN ILLUMINATION
    type_sfero,
    type_kubo,
    type_cilindro,
    
    type_ebeno = 0x0080,

    type_kunigajo_2 = 0x0100,
    type_kunigajo_3,
    type_kunigajo_4,            // YET NOT IMPLEMENTED
    type_komunajo_2,
    type_komunajo_3,
    type_komunajo_4,            // YET NOT IMPLEMENTED
    type_komplemento,
    type_glata_kunigajo_2,
    type_glata_komunajo_2,

    type_movo = 0x0200,
    type_rotacioX,
    type_rotacioY,
    type_rotacioZ,
    type_rotacioQ,
    type_senfina_ripeto
};

struct bazo {
    enum object_type type;
    raw_byte data[ PRIMITIVE_PAYLOAD ];
    dist_func dist;
    norm_func norm;
    explicit bazo( enum object_type _type = type_nenio ) : type( _type ), data{}, dist( nullptr ), norm( nullptr ) {
        memset( data, 0x00, PRIMITIVE_PAYLOAD );
    }
};

template<typename ARG>
bazo create( enum object_type _type, ARG *arg ) {
    primitives::bazo NEW( _type );
    memcpy( NEW.data, arg, sizeof( ARG ) );
    return NEW;
}

};

namespace null {

class raymarching {
protected:
    size_t Width, Height;
    
    std::string SceneName;
    std::vector<primitives::bazo> Primitives_h;
    size_t PrimitivesNum;
    
    void *Primitives_d;
    void *Rays_d;
    void *Info_d;
public:
    /****************************************************************************
     * @brief   Initializes environment for specific compute method             *
     * @param   width   : specify a width of new window                         *
     * @param   height  : specify a height of new window                        *
     * @return  0 on success, ERR_CODE>0 otherwise                              *
     ****************************************************************************/
    virtual int Init( rays_Init_args ) = 0;
    
    /****************************************************************************
     * @brief   Compute one frame of a given scene                              *
     * @return  Time of processing in milliseconds                              *
     ****************************************************************************/
    virtual int Process( rays_Process_args ) = 0;
    
    /****************************************************************************
     * @brief   Performs quit hooks to clear environment for new compute method *
     * @return  The last ERR_CODE of computations or of quit hooks              *
     ****************************************************************************/
    virtual int Quit( rays_Quit_args ) = 0;
    
    
    virtual int SetInfo( rays_SetInfo_args ) = 0;
    
    virtual int SetTexture( rays_SetTexture_args ) = 0;
    virtual int UnsetTexture( rays_UnsetInfo_args ) = 0;
    
    virtual int SetPrimitives( rays_SetPrimitives_args ) = 0;
    
    virtual int SetRays( rays_SetRays_args ) = 0;
    
    
    inline void SetSceneName( const char* name ) {
        SceneName = name;
    }
    inline void ReservePrimitives( size_t new_capacity ) {
        Primitives_h.reserve( new_capacity );
        PrimitivesNum = 0;
    }
    inline void AddPrimitive( primitives::bazo new_primitive ) {
        Primitives_h.push_back( new_primitive );
        PrimitivesNum++;
    }
    inline void ClearPrimitives() {
        Primitives_h.clear();
        PrimitivesNum = 0;
    }
};

}