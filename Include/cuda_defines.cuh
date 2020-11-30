#pragma once

#define CUDA_ARCH_COMPILE_FLAG()    ( std::string("-arch=compute_") + __CUDA_ARCH__ ).c_str()

// Windows library getter
#define CUDA_LIB_PATH_WIN(__LIB__)  ( std::string(__CUDA_DIR__) + STRINGIFY(__LIB__) + ".lib" ).c_str()
#define CUDA_LIB_PATH_UNX(__LIB__)  ( std::string(__CUDA_DIR__) + STRINGIFY(__LIB__) + ".a" ).c_str()

#define CUDA_RAYS_STREAM_NUM        1
#define CUDA_RAYS_DEFAULT_STREAM    0

#define CUDA_RAYS_EVENT_NUM         2

#ifdef _DEBUG

#define CUDA_CHECK(__ERROR__) {                                                         \
    CUresult err = ( __ERROR__ );                                                       \
    if ( err ) {                                                                        \
        const char *ErrorName;                                                          \
        cuGetErrorName(err,&ErrorName);                                                 \
        std::cout <<                                                                    \
            ErrorName << std::endl <<                                                   \
            "\t: at line " << __LINE__ << std::endl <<                                  \
            "\t: in file " << __FILE__ << std::endl << std::endl;                       \
    }                                                                                   \
}

#define NVRTC_CHECK(__ERROR__) {                                                        \
    nvrtcResult err = ( __ERROR__ );                                                    \
    if ( err ) {                                                                        \
        const char *ErrorName = nvrtcGetErrorString(err);                               \
        std::cout <<                                                                    \
            ErrorName << std::endl <<                                                   \
            "\t: at line " << __LINE__ << std::endl <<                                  \
            "\t: in file " << __FILE__ << std::endl << std::endl;                       \
    }                                                                                   \
}

#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(__ERROR__) (void)(__ERROR__)
#endif

#ifndef NVRTC_CHECK
#define NVRTC_CHECK(__ERROR__) (void)(__ERROR__)
#endif

#define length_2(x,y)       (hypotf((x),(y)))
#define length_3(x,y,z)     (norm3df((x),(y),(z)))

#define r_length_2(x,y)     (rhypotf((x),(y)))
#define r_length_3(x,y,z)   (rnorm3df((x),(y),(z)))

#define CUDA_RAYS_COORD_nD(c,n)  ((blockIdx. c) * (RAYS_BLOCK_##n##D_##c) + (threadIdx. c))