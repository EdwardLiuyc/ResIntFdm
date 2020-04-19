cmake_minimum_required(VERSION 3.1)
# Enable C++11
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall \
  -Wextra -fPIC \
  -Wno-sign-compare \
  -Wno-unused-parameter \
  -Wno-unused-but-set-variable \
  -Wno-unused-variable ")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
include_directories(${PROJECT_SOURCE_DIR})

################### Dependencies ############################
### opencv ###
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
link_libraries(${OpenCV_LIBRARIES})

### eigen ###
find_package(Eigen3 3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message( STATUS "Found Eigen3 version: ${EIGEN3_VERSION}")

### Boost ###
find_package(Boost REQUIRED 
  filesystem)
link_libraries(${Boost_LIBRARIES})

################### Optional Dependencies ####################
### TBB ###
option(USE_TBB "Enable TBB?" OFF) #set to OFF to disable
if(USE_TBB)
  find_package(TBB)
  if(TBB_FOUND)
    message(STATUS "TBB version: ${TBB_INTERFACE_VERSION}")
    include_directories(${TBB_INCLUDE_DIRS})
    list(APPEND require_libs ${TBB_LIBRARIES})
    add_definitions(-D_USE_TBB_)
  endif(TBB_FOUND)
endif(USE_TBB)

### Cuda ###
option(ENABLE_CUDA "Enable Cuda?" ON)
if(ENABLE_CUDA)
  find_package(CUDA)
  if (CUDA_FOUND)
    message(STATUS "CUDA found. CudaSift will be built.")
    add_subdirectory(third_parties/CudaSift)
    add_definitions("-D_USE_CUDA_")
  else(CUDA_FOUND)
    set(ENABLE_CUDA OFF)
  endif(CUDA_FOUND)
endif(ENABLE_CUDA)
### glog ### 
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

################### Build Targets ########################
file(GLOB srcs "*.cc" ) 
add_executable(fdm ${srcs})
target_link_libraries(fdm ${GLOG_LIBRARIES})

if(ENABLE_CUDA)
  target_link_libraries(fdm cuda_sift)
endif(ENABLE_CUDA)

enable_testing()
add_subdirectory(third_parties/googletest)
add_subdirectory(tests)

