cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(OT_libtorch)  #src代表项目名
 
SET(CMAKE_C_COMPILER g++)
# add_definitions(--std=c++14) #注意自己的c++版本，之前写的--std=c++11 报错，根据电脑配置定


find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 指定libTorch位置
set(Torch_DIR /home/mjy/toytools/libtorch/libtorch-shared-with-deps-1.10.2+cu113/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
message(STATUS "Torch library status:")
message(STATUS "    version: ${TORCH_VERSION}")
message(STATUS "    libraries: ${TORCH_LIBS}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}")
message(STATUS "    torch lib : ${TORCH_LIBRARIES} ")


 

message(STATUS "    torch lib : ${TORCH_LIBRARIES} ")
 
# 包含头文件include
# include_directories(${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
 

add_executable(fast_ot fast_ot.cpp)
target_link_libraries(fast_ot ${TORCH_LIBRARIES} ${PCL_LIBRARIES})


