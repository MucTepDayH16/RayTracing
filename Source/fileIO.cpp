#include "../Include/fileIO.h"
#include <SDL.h>

namespace IO {

using namespace std;

string      read_source( const string &file_name ) {
    ifstream fin( file_name.c_str() );
    string file_content((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
    fin.close();
    return file_content;
}

void        write_source( const string &file_name, const string &to_write ) {
    ofstream fout( file_name.c_str(), ios::trunc );
    fout.write( to_write.c_str(), to_write.size() );
    fout.close();
}

char*       read_binary( const std::string &file_name ) {
    ifstream fin( file_name.c_str(), ios::binary | ios::ate );
    size_t len = fin.tellg();
    fin.seekg( 0, ios::beg );
    
    char *src = new char [ len ];
    fin.read( src, len );
    fin.close();
    return src;
}

void        write_binary( const std::string &file_name, const void *src, const size_t &len ) {
    ofstream fout( file_name.c_str(), ios::binary | ios::trunc );
    fout.write( reinterpret_cast<const char*>(src), len );
    fout.close();
}

#define DATA    reinterpret_cast<write_binary_args*>(data)

struct write_binary_args {
    const std::string *file_name;
    const void *src;
    size_t len;
};

static int  write_binary_th( void *data ) {
    ofstream fout( DATA->file_name->c_str(), ios::binary | ios::trunc );
    fout.write( reinterpret_cast<const char*>(DATA->src), DATA->len );
    fout.close();
    
    delete DATA->file_name;
    return 0;
}

void        write_binary_nowait( const std::string *file_name, const void *src, size_t len ) {
    write_binary_args args = {
            .file_name = file_name,
            .src = src,
            .len = len,
    };
    SDL_Thread  *write_th = SDL_CreateThread( write_binary_th, "write_binary", &args );
}

};