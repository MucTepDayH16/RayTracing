#include "rays.h"
#include <fstream>

namespace raymarching {

int PrimitivesI( std::vector< primitives::bazo > &primitives, const std::string &file_name ) {
    std::fstream fin;
    fin.open( file_name.c_str(), std::ios::in | std::ios::binary );
    char *buffer = new char[ primitives.capacity() * sizeof primitives::bazo ];
    fin.read( buffer, primitives.capacity() * sizeof primitives::bazo );
    size_t num = fin.gcount() / sizeof primitives::bazo;
    fin.close();
    primitives.resize( num );
    memcpy( primitives.data(), buffer, num * sizeof primitives::bazo );
    delete buffer;
    return 0;
}

int PrimitivesO( const std::vector< primitives::bazo > &primitives, const std::string &file_name ) {
    std::fstream fout;
    fout.open( file_name.c_str(), std::ios::out | std::ios::binary );
    fout.write( ( char* ) ( void* ) primitives.data(), primitives.size() * sizeof primitives::bazo );
    fout.close();
    return 0;
}

};