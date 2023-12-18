# if(NOT TARGET inotify-cpp::inotify-cpp)
#     find_package(inotify-cpp CONFIG REQUIRED)
# endif()

find_package(CURL REQUIRED)
# Target
add_executable(coordinator coordinator.cpp ${PROJECT_SOURCE_DIR}/util/easylogging++.cc)
target_include_directories(coordinator PRIVATE ${PROJECT_SOURCE_DIR}/util)
target_link_libraries(coordinator PRIVATE inotify-cpp::inotify-cpp json11 curl)