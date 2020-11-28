#include "../Include/sha3.h"

#define ROT_64(__VAL__,__ROT__) uint64_t((__VAL__) << (__ROT__) | (__VAL__) >> (64-(__ROT__)) )


namespace sha3 {

static inline void keccak_permute( uint64_t a[ 5 ][ 5 ] ) {
    const uint64_t
    R[ 24 ] = {
            0x0000000000000001uL, 0x0000000000008082uL, 0x800000000000808auL,
            0x8000000080008000uL, 0x000000000000808buL, 0x0000000080000001uL,
            0x8000000080008081uL, 0x8000000000008009uL, 0x000000000000008auL,
            0x0000000000000088uL, 0x0000000080008009uL, 0x000000008000000auL,
            0x000000008000808buL, 0x800000000000008buL, 0x8000000000008089uL,
            0x8000000000008003uL, 0x8000000000008002uL, 0x8000000000000080uL,
            0x000000000000800auL, 0x800000008000000auL, 0x8000000080008081uL,
            0x8000000000008080uL, 0x0000000080000001uL, 0x8000000080008008uL,
    };
    uint64_t
        C[ 5 ] = { 0uL, 0uL, 0uL, 0uL, 0uL },
        D = 0uL;
    
    for ( uint8_t i = 0; i < 24; ++i ) {
        
        for ( uint8_t x = 0; x < 5; ++x )
            C[ x ] = a[ x ][ 0 ] ^ a[ x ][ 1 ] ^ a[ x ][ 2 ] ^ a[ x ][ 3 ] ^ a[ x ][ 4 ];
        
        for ( uint8_t x = 0; x < 5; ++x ) {
            D = C[ ( x + 4 ) % 5 ] ^ ROT_64( C[ ( x + 1 ) % 5 ], 1uL );
            
            a[ x ][ 0 ] ^= D;
            a[ x ][ 1 ] ^= D;
            a[ x ][ 2 ] ^= D;
            a[ x ][ 3 ] ^= D;
            a[ x ][ 4 ] ^= D;
        }
        
        uint8_t XY = 0x10u;
        D = a[ XY >> 4u ][ XY & 0x0fu ];
        for ( uint8_t t = 0; t < 24; ++t ) {
            XY =    ( XY << 4u ) |
                    ( ( ( ( XY >> 3u ) + ( XY & 0x0fu ) * 3u ) % 5u ) & 0x0fu );
            C[ 0 ] = a[ XY >> 4u ][ XY & 0x0fu ];
            C[ 1 ] = ( ( ( t + 1u ) * ( t + 2u ) ) >> 1u ) & 0x3fu;
            a[ XY >> 4u ][ XY & 0x0fu ] = ROT_64( D, C[ 1 ] );
            D = C[ 0 ];
        }
        
        for ( uint8_t y = 0; y < 5; ++y ) {
            C[ 0 ] = a[ 0 ][ y ];
            C[ 1 ] = a[ 1 ][ y ];
            C[ 2 ] = a[ 2 ][ y ];
            C[ 3 ] = a[ 3 ][ y ];
            C[ 4 ] = a[ 4 ][ y ];
            
            a[ 0 ][ y ] ^= ( ~C[ 1 ] & C[ 2 ] );
            a[ 1 ][ y ] ^= ( ~C[ 2 ] & C[ 3 ] );
            a[ 2 ][ y ] ^= ( ~C[ 3 ] & C[ 4 ] );
            a[ 3 ][ y ] ^= ( ~C[ 4 ] & C[ 0 ] );
            a[ 4 ][ y ] ^= ( ~C[ 0 ] & C[ 1 ] );
        }
        
        a[ 0 ][ 0 ] ^= R[ i ];
    }
}

#define BITRATE 576uL
#define WIDTH   64uL
#define BLOCK   ( BITRATE / WIDTH )
#define PITCH   ( BLOCK << 3u )

#define LENGTH  ( BITRATE >> 3u )

uint8_t* hash_512( std::string source ) {
    uint64_t state[ 5 ][ 5 ] = {
            { 0uL, 0uL, 0uL, 0uL, 0uL },
            { 0uL, 0uL, 0uL, 0uL, 0uL },
            { 0uL, 0uL, 0uL, 0uL, 0uL },
            { 0uL, 0uL, 0uL, 0uL, 0uL },
            { 0uL, 0uL, 0uL, 0uL, 0uL },
    };
    
    source.reserve( ( 1 + ( source.size() / LENGTH ) ) * LENGTH );
    size_t  pad = LENGTH - ( source.size() % LENGTH );
    if ( pad == 1 ) {
        source.push_back( 0x86 );
    } else {
        source.push_back( 0x06 );
        while ( pad-- > 2 ) {
            source.push_back( 0x00 );
        }
        source.push_back( 0x80 );
    }
    
    size_t          i, j;
    uint64_t        num;
    for ( i = 0; i < source.size(); i += PITCH ) {
        for ( j = 0; j < BLOCK; ++j ) {
            memcpy( &num, source.c_str() + i + ( j << 3u ), 8 );
            state[ j % 5 ][ j / 5 ] ^= num;
        }
        keccak_permute( state );
    }
    
    uint8_t *ret_hash = new uint8_t [ 64 ];
    for ( i = 0; i < 8; ++i ) {
        memcpy( ret_hash + ( i << 3u ), &state[ i % 5 ][ i / 5 ], 8 );
    }
    
    return  ret_hash;
}

};