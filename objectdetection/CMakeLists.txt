cmake_minimum_required(VERSION 3.18)
project(ObjectDetection LANGUAGES CXX)

message(STATUS "CMake generator: ${CMAKE_GENERATOR}, platform: ${CMAKE_GENERATOR_PLATFORM}")
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "检测到当前是 32 位(Win32) 构建，请改用 x64 Kit 或使用 -A x64 重新配置。")
endif()

option(BUILD_EXE "Build the example executable" ON)
option(BUILD_DLL_C_API "Build the C API DLL" ON)
option(BUILD_PYBIND "Build the Python extension via pybind11" OFF)
option(USE_FETCHCONTENT_PYBIND11 "Fetch pybind11 if not found" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Third-party default roots (inside repository)
set(THIRDPARTY_ROOT "F:/02-nm/hjj-2/windows-x64/ObjectDetection/objectdetection/3rdPart" CACHE PATH "Root path for third-party dependencies")

# OpenCV
# 首选使用官方包配置；如不兼容则回退到手动链接 OpenCV 4.5.4 (world)
set(OpenCV_DIR "F:/02-nm/hjj-2/windows-x64/ObjectDetection/objectdetection/3rdPart/opencv454-win/opencv/build")
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs highgui dnn)
if(OpenCV_FOUND)
  message(STATUS "OpenCV version: ${OpenCV_VERSION}")
else()
  message(WARNING "OpenCVConfig.cmake 不兼容当前构建，回退到手动链接 OpenCV 4.5.4 (world)")
  set(OPENCV_ROOT "${THIRDPARTY_ROOT}/opencv454-win/opencv")
  set(_OPENCV_INC_LIST
    "${OPENCV_ROOT}/build/include"
    "${OPENCV_ROOT}/build/include/opencv4"
    "${OPENCV_ROOT}/include"
  )
  set(OPENCV_INCLUDE_DIRS "")
  foreach(_p IN LISTS _OPENCV_INC_LIST)
    if(EXISTS "${_p}")
      list(APPEND OPENCV_INCLUDE_DIRS "${_p}")
    endif()
  endforeach()
  set(_OPENCV_LIB_DIR_CANDIDATES
    "${OPENCV_ROOT}/build/x64/vc17/lib"
    "${OPENCV_ROOT}/build/x64/vc16/lib"
    "${OPENCV_ROOT}/build/x64/vc15/lib"
    "${OPENCV_ROOT}/build/x64/vc14/lib"
  )
  set(OPENCV_LIB_DIR "")
  foreach(_p IN LISTS _OPENCV_LIB_DIR_CANDIDATES)
    if(EXISTS "${_p}")
      set(OPENCV_LIB_DIR "${_p}")
      break()
    endif()
  endforeach()
  if(NOT OPENCV_LIB_DIR)
    message(FATAL_ERROR "未找到可用的 OpenCV lib 目录，请检查 ${OPENCV_ROOT}/build/x64/vcXX/lib")
  endif()
  # OpenCV 4.5.4 预编译 world 库名称
  set(OPENCV_LIB_RELEASE opencv_world454)
  set(OPENCV_LIB_DEBUG opencv_world454d)
endif()

# ONNX Runtime discovery
set(ONNXRUNTIME_ROOT "${THIRDPARTY_ROOT}/onnxruntime-win-x64-1.19.0" CACHE PATH "Root of ONNX Runtime (should contain include/ and lib/)")

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "Could not find ONNX Runtime. Set ONNXRUNTIME_ROOT to your install path.")
else()
  message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
  message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB}")
endif()

# Validate ORT header presence to avoid cryptic compile errors
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "onnxruntime_cxx_api.h not found at: ${ONNXRUNTIME_INCLUDE_DIR}. Please check ONNXRUNTIME_ROOT or ONNXRUNTIME_INCLUDE_DIR.")
endif()

# Optional DLL to copy after build
if(WIN32)
  # Try to auto-detect DLL next to provided ONNXRUNTIME_ROOT
  if(NOT DEFINED ONNXRUNTIME_DLL)
    set(_ORT_DLL_GUESS "${ONNXRUNTIME_ROOT}/onnxruntime.dll")
    if(EXISTS "${_ORT_DLL_GUESS}")
      set(ONNXRUNTIME_DLL "${_ORT_DLL_GUESS}" CACHE FILEPATH "Path to onnxruntime.dll for post-build copy (Windows)")
    else()
      set(ONNXRUNTIME_DLL "" CACHE FILEPATH "Path to onnxruntime.dll for post-build copy (Windows)")
    endif()
  endif()
else()
  set(ONNXRUNTIME_DLL "" CACHE FILEPATH "Path to onnxruntime.dll for post-build copy (Windows)")
endif()

add_library(yolo_core STATIC
  YOLO.cpp
  YOLO.h
)
target_include_directories(yolo_core
  PUBLIC
    $<$<BOOL:${OpenCV_FOUND}>:${OpenCV_INCLUDE_DIRS}>
    $<$<NOT:$<BOOL:${OpenCV_FOUND}>>:${OPENCV_INCLUDE_DIRS}>
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}
)
if(NOT OpenCV_FOUND)
  target_link_directories(yolo_core PUBLIC ${OPENCV_LIB_DIR})
endif()
# 在未找到 OpenCV 包配置时，基于配置选择 world 库（Debug/非Debug）
if(NOT OpenCV_FOUND)
  set(OPENCV_WORLD_LIB
    $<$<CONFIG:Debug>:${OPENCV_LIB_DEBUG}>
    $<$<NOT:$<CONFIG:Debug>>:${OPENCV_LIB_RELEASE}>
  )
endif()
target_link_libraries(yolo_core
  PUBLIC
    ${ONNXRUNTIME_LIB}
    $<$<BOOL:${OpenCV_FOUND}>:${OpenCV_LIBS}>
    $<$<NOT:$<BOOL:${OpenCV_FOUND}>>:${OPENCV_WORLD_LIB}>
)

if(BUILD_EXE)
  add_executable(objectdetect_app
    ObejectDetect.cpp
  )
  target_include_directories(objectdetect_app PRIVATE ${CMAKE_CURRENT_LIST_DIR})
  # 显式链接 OpenCV/ORT，避免由于跨目标使用而产生未解析符号
  if(OpenCV_FOUND)
    target_link_libraries(objectdetect_app PRIVATE yolo_core ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})
  else()
    target_link_directories(objectdetect_app PRIVATE ${OPENCV_LIB_DIR})
    set(OPENCV_WORLD_LIB_APP
      $<$<CONFIG:Debug>:${OPENCV_LIB_DEBUG}>
      $<$<NOT:$<CONFIG:Debug>>:${OPENCV_LIB_RELEASE}>
    )
    target_link_libraries(objectdetect_app PRIVATE yolo_core ${OPENCV_WORLD_LIB_APP} ${ONNXRUNTIME_LIB})
  endif()
  if(WIN32 AND EXISTS "${ONNXRUNTIME_DLL}")
    add_custom_command(TARGET objectdetect_app POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_DLL}" "$<TARGET_FILE_DIR:objectdetect_app>/"
      COMMENT "Copying onnxruntime.dll next to objectdetect_app"
    )
  endif()
endif()

if(BUILD_DLL_C_API)
  add_library(yolo_capi SHARED
    c_api.cpp
    c_api.h
  )
  target_include_directories(yolo_capi PUBLIC ${CMAKE_CURRENT_LIST_DIR})
  if(OpenCV_FOUND)
    target_link_libraries(yolo_capi PRIVATE yolo_core ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})
  else()
    target_link_directories(yolo_capi PRIVATE ${OPENCV_LIB_DIR})
    set(OPENCV_WORLD_LIB_DLL
      $<$<CONFIG:Debug>:${OPENCV_LIB_DEBUG}>
      $<$<NOT:$<CONFIG:Debug>>:${OPENCV_LIB_RELEASE}>
    )
    target_link_libraries(yolo_capi PRIVATE yolo_core ${OPENCV_WORLD_LIB_DLL} ${ONNXRUNTIME_LIB})
  endif()
  set_target_properties(yolo_capi PROPERTIES OUTPUT_NAME "yolo_capi")
  if(WIN32 AND EXISTS "${ONNXRUNTIME_DLL}")
    add_custom_command(TARGET yolo_capi POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_DLL}" "$<TARGET_FILE_DIR:yolo_capi>/"
      COMMENT "Copying onnxruntime.dll next to yolo_capi"
    )
  endif()
endif()

if(BUILD_PYBIND)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND AND USE_FETCHCONTENT_PYBIND11)
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
    # 使用 FetchContent 时，直接将其视为已找到（或检查目标是否存在）
    set(pybind11_FOUND TRUE)
  endif()
  if(NOT pybind11_FOUND AND NOT TARGET pybind11::module)
    message(FATAL_ERROR "pybind11 not found. Either install it (pip install pybind11) and make sure CMake can find it, or enable USE_FETCHCONTENT_PYBIND11.")
  endif()
  pybind11_add_module(yolort MODULE
    yolo_bindings.cpp
  )
  target_include_directories(yolort PRIVATE ${CMAKE_CURRENT_LIST_DIR})
  target_link_libraries(yolort PRIVATE yolo_core)
  if(WIN32 AND EXISTS "${ONNXRUNTIME_DLL}")
    add_custom_command(TARGET yolort POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_DLL}" "$<TARGET_FILE_DIR:yolort>/"
      COMMENT "Copying onnxruntime.dll next to yolort"
    )
  endif()
endif()


