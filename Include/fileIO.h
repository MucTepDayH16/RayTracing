#pragma once
#include <fstream>

namespace IO {

std::string read_source( const std::string &file_name );
void        write_source( const std::string &file_name, const std::string &to_write );

};