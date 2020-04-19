find_package(OpenCV REQUIRED)

include(${PROJECT_SOURCE_DIR}/cmake_modules/SelectCudaComputeArch.cmake)
CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${CUDA_ARCHS})
# Do not show warnings if the architectures are deprecated.
list(APPEND CUDA_NVCC_FLAGS "${CUDA_ARCH_FLAGS};-D_FORCE_INLINES;-Wno-deprecated-gpu-targets")
message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
                " archs: ${CUDA_ARCH_FLAGS_readable})")

list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-DVERBOSE_NOT")

if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
    list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-O2;-DVERBOSE") 
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin;/usr/bin/gcc;-O2") 
  endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR} )
set(cuda_sources
  cudaImage.cu  
  cudaImage.h  
  cudaSiftH.cu 
  cudaSiftH.h  
  matching.cu  
  cudaSiftD.h  
  cudaSift.h  
  cudautils.h
  geomFuncs.h 
  geomFuncs.cpp  
)
cuda_add_library(cuda_sift ${cuda_sources})

set(sources
  mainSift.cpp  
)

cuda_add_executable(cudasift ${sources} OPTIONS ${CUDA_ARCH_FLAGS})
set_target_properties(cudasift PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)
target_link_libraries(cudasift cuda_sift ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})
