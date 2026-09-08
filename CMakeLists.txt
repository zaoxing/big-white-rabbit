cmake_minimum_required(VERSION 3.21)
project(freetoken_mac LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT APPLE)
    message(FATAL_ERROR "FreeToken-Mac targets Apple Silicon; build llama.cpp directly on other platforms.")
endif()

# --- llama.cpp / ggml with the Metal backend ------------------------------------------
# Static link llama/ggml straight into _freetoken_metal.so. A shared build leaves the
# extension chasing @rpath/libllama.0.dylib, which the wheel does not ship -- and on a
# machine with conda on PATH that rpath resolves into the conda prefix and fails to load.
# One self-contained .so avoids shipping dylibs and rewriting install_name entirely.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# GGML_METAL_EMBED_LIBRARY (defaults to ON alongside GGML_METAL) embeds the shader SOURCE
# and compiles it at runtime via newLibraryWithSource. That is deliberately left ON: the
# OFF path shells out to `xcrun metal`/`metallib` at build time, which requires the
# separately-downloaded Metal offline toolchain that is not present on every dev machine.
set(GGML_METAL              ON  CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(GGML_BLAS               OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE         ON  CACHE BOOL "" FORCE)

# We link libllama only; the CLI tools/tests/server are not part of this build.
# (Build them ad hoc from third_party/llama.cpp for Phase 0 verification.)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# --- the extension module -------------------------------------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_freetoken_metal
    csrc/module.cpp
    csrc/model.cpp
    csrc/context.cpp
)

target_link_libraries(_freetoken_metal PRIVATE llama ggml)
target_compile_options(_freetoken_metal PRIVATE -Wall -Wextra)

install(TARGETS _freetoken_metal DESTINATION freetoken_mac)
