#pragma once

#include "defines.h"
#include "types.h"

namespace primitives {

CREATE_OBJECT_TYPE_DESCRIPTION( portanta_sfero, struct { counter o; point t; scalar r; } )
CREATE_OBJECT_TYPE_DESCRIPTION( sfero, struct { scalar r; } )
CREATE_OBJECT_TYPE_DESCRIPTION( kubo, struct { point b; } )
CREATE_OBJECT_TYPE_DESCRIPTION( cilindro, struct { scalar r; scalar h; } )

CREATE_OBJECT_TYPE_DESCRIPTION( ebeno, struct { point n; } )

CREATE_OBJECT_TYPE_DESCRIPTION( kunigajo_2, struct { counter o[ 2 ]; } )
CREATE_OBJECT_TYPE_DESCRIPTION( kunigajo_3, struct { counter o[ 3 ]; } )
CREATE_OBJECT_TYPE_DESCRIPTION( kunigajo_4, struct { counter o[ 4 ]; } )
CREATE_OBJECT_TYPE_DESCRIPTION( komunajo_2, struct { counter o[ 2 ]; } )
CREATE_OBJECT_TYPE_DESCRIPTION( komunajo_3, struct { counter o[ 3 ]; } )
CREATE_OBJECT_TYPE_DESCRIPTION( komunajo_4, struct { counter o[ 4 ]; } )
CREATE_OBJECT_TYPE_DESCRIPTION( komplemento, struct { counter o; } )
CREATE_OBJECT_TYPE_DESCRIPTION( glata_kunigajo_2, struct { counter o[ 2 ]; scalar k; } )
CREATE_OBJECT_TYPE_DESCRIPTION( glata_komunajo_2, struct { counter o[ 2 ]; scalar k; } )

CREATE_OBJECT_TYPE_DESCRIPTION( movo, struct { counter o; point t; } )
CREATE_OBJECT_TYPE_DESCRIPTION( rotacioX, struct { counter o; scalar cos_phi; scalar sin_phi; } )
CREATE_OBJECT_TYPE_DESCRIPTION( rotacioY, struct { counter o; scalar cos_phi; scalar sin_phi; } )
CREATE_OBJECT_TYPE_DESCRIPTION( rotacioZ, struct { counter o; scalar cos_phi; scalar sin_phi; } )
CREATE_OBJECT_TYPE_DESCRIPTION( rotacioQ, struct { counter o; scalar q_w; point q; } )
CREATE_OBJECT_TYPE_DESCRIPTION( senfina_ripeto, struct { counter o; point a; } )

};