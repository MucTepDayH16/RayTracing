#include "rays.h"

namespace cuda {

static size_t _streams_count;
static cudaStream_t *_stream;

bool Init( size_t StreamsCount ) {
    _streams_count = StreamsCount;
    _stream = new cudaStream_t[ _streams_count ];

    for ( size_t n = 0; n < _streams_count; ++n ) {
        cudaStreamCreate( _stream + n );
    }

    return true;
}

cudaStream_t Stream( size_t i ) {
    return _stream[ i % _streams_count ];
}

};