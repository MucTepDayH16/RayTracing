#pragma once

#include "objects_list.h"
#include "fileIO.h"

#include <vector>
#include <string>
#include <iostream>

#include <SDL.h>
#include <SDL_opengl.h>

namespace null {

class raymarching {
protected:
    size_t Width, Height;
    
    std::string SceneName;
    std::vector<primitives::bazo> Primitives_h;
    size_t PrimitivesNum;
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