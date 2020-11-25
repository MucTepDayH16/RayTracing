#include "../Include/fileIO.h"

namespace IO {

using namespace std;

string read_source( const char *file_name ) {
    ifstream fin( file_name );
    string file_content((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
    fin.close();
    return file_content;
}

void    write_source( const char *file_name, const string &to_write ) {
    ofstream fout( file_name, ios_base::trunc );
    fout.write( to_write.c_str(), to_write.size() );
    fout.close();
}

};