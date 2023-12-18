cmake_minimum_required(VERSION 3.10)
project(coordinator)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(ext/json11)
add_subdirectory(ext/inotify-cpp)
add_subdirectory(src)