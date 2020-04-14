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

### opencv ###
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
link_libraries(${OpenCV_LIBRARIES})

find_package(Eigen3 3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message( STATUS 
  "Found Eigen in dir: ${EIGEN3_INCLUDE_DIR} version: ${EIGEN3_VERSION}")

enable_testing()
add_subdirectory(third_parties/googletest)
add_subdirectory(tests)

# add_executable(meb meb.cc meb_test.cc)