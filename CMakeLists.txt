cmake_minimum_required(VERSION 3.20)
project(tglab CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)

# ---------------------------------------------------------------------------
# Dear ImGui (docking branch) + win32/dx12 backends
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_win32.cpp
    ${IMGUI_DIR}/backends/imgui_impl_dx12.cpp
)
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(imgui PUBLIC d3d12 dxgi dwmapi)

# ---------------------------------------------------------------------------
# Core library: images, params, algorithm base, registry, pipeline, script
add_library(tglab_core STATIC
    src/core/image.cpp
    src/core/image_io.cpp
    src/core/param.cpp
    src/core/algorithm.cpp
    src/core/pipeline.cpp
    src/script/lexer.cpp
    src/script/parser.cpp
    src/script/interp.cpp
    src/gpu/descriptor_heap.cpp
    src/gpu/device.cpp
    src/gpu/texture.cpp
)
target_include_directories(tglab_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party/stb
)
target_link_libraries(tglab_core PUBLIC imgui d3d12 dxgi)

# ---------------------------------------------------------------------------
# Algorithms.
#
# MUST be an OBJECT library. Algorithms self-register via a static initializer
# and nothing references their symbols, so a STATIC library would be stripped
# by the linker and they would silently vanish from the registry.
add_library(tglab_algorithms OBJECT
    src/algorithms/brightness.cpp
    src/algorithms/grayscale.cpp
    src/algorithms/gaussian_blur.cpp
    src/algorithms/sobel.cpp
    src/algorithms/canny_stages.cpp
    src/algorithms/canny.cpp
)
target_link_libraries(tglab_algorithms PUBLIC tglab_core)

# ---------------------------------------------------------------------------
add_executable(tglab
    src/app/main.cpp
    src/app/image_view.cpp
    src/app/file_watch.cpp
    $<TARGET_OBJECTS:tglab_algorithms>
)
target_link_libraries(tglab PRIVATE tglab_core imgui d3d12 dxgi)

if(MSVC)
    target_compile_options(tglab PRIVATE /W4 /permissive-)
    target_compile_options(tglab_core PRIVATE /W4 /permissive-)
    target_compile_definitions(tglab PRIVATE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_definitions(tglab_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_definitions(tglab_algorithms PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Run from the source dir so scripts/ and assets/ resolve.
set_target_properties(tglab PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# ---------------------------------------------------------------------------
# Headless checks for the script spine (no window required).
add_executable(tglab_tests
    tests/test_script.cpp
    $<TARGET_OBJECTS:tglab_algorithms>
)
target_link_libraries(tglab_tests PRIVATE tglab_core imgui d3d12 dxgi)
if(MSVC)
    target_compile_definitions(tglab_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

enable_testing()
add_test(NAME script_spine COMMAND tglab_tests)






