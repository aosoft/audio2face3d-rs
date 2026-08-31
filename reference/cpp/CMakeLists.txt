cmake_minimum_required(VERSION 3.25)
project(audio2face3d_cpp_reference LANGUAGES CXX)

if("$ENV{AUDIO2FACE_SDK_ROOT}" STREQUAL "")
  message(FATAL_ERROR "AUDIO2FACE_SDK_ROOT environment variable must point to the NVIDIA Audio2Face-3D-SDK checkout")
endif()

file(TO_CMAKE_PATH "$ENV{AUDIO2FACE_SDK_ROOT}" AUDIO2FACE_SDK_ROOT)
set(AUDIO2X_SDK_DIR "${AUDIO2FACE_SDK_ROOT}/_build/release/audio2x-sdk")
if(NOT EXISTS "${AUDIO2X_SDK_DIR}/lib/audio2x.lib")
  message(FATAL_ERROR "Built audio2x.lib was not found under ${AUDIO2X_SDK_DIR}")
endif()

add_executable(audio2face3d-cpp-reference main.cpp)
target_compile_features(audio2face3d-cpp-reference PRIVATE cxx_std_20)
target_include_directories(audio2face3d-cpp-reference PRIVATE "${AUDIO2X_SDK_DIR}/include")
target_link_libraries(audio2face3d-cpp-reference PRIVATE
  "${AUDIO2X_SDK_DIR}/lib/audio2x.lib"
  bcrypt)

if(MSVC)
  target_compile_options(audio2face3d-cpp-reference PRIVATE /W4 /EHsc /MD /GL-)
  # Avoid the incremental/LTCG state involved in the SDK aggregate-link LNK1000.
  target_link_options(audio2face3d-cpp-reference PRIVATE /INCREMENTAL:NO /LTCG:OFF /DEBUG:NONE)
endif()

add_custom_command(TARGET audio2face3d-cpp-reference POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${AUDIO2X_SDK_DIR}/bin/audio2x.dll"
    "$<TARGET_FILE_DIR:audio2face3d-cpp-reference>/audio2x.dll")
