#pragma once
#include <fstream>

namespace IO {

std::string read_source( const char *file_name );
void        write_source( const char *file_name, const std::string &to_write );

};