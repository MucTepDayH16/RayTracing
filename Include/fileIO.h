#pragma once
#include <fstream>
#include <vector>

namespace IO {

std::string read_source( const std::string &file_name );
void        write_source( const std::string &file_name, const std::string &to_write );

std::vector<char>
            read_binary( const std::string &file_name );
void        write_binary( const std::string &file_name, const void *src, const size_t &len );
void        write_binary_nowait( const std::string *file_name, const void *src, size_t len );

};