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
# LibRaw (camera raw decoding). Built from source; see cmake/libraw.cmake.
include(${CMAKE_SOURCE_DIR}/cmake/libraw.cmake)

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
    src/core/worker.cpp
    src/core/compare.cpp
    src/core/image_loader.cpp
    src/algo_util/histogram.cpp
    src/algo_util/pixel_buffer.cpp
    src/core/exif.cpp
    src/core/image_stats.cpp
    src/core/raw_io.cpp
    src/script/lexer.cpp
    src/script/parser.cpp
    src/script/interp.cpp
    src/gpu/descriptor_heap.cpp
    src/gpu/device.cpp
    src/gpu/texture.cpp
    src/gpu/shader.cpp
    src/gpu/gpu_image.cpp
    src/gpu/compute.cpp
)
target_include_directories(tglab_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party/stb
)
target_link_libraries(tglab_core PUBLIC imgui libraw d3d12 dxgi dxcompiler)

# ---------------------------------------------------------------------------
# Algorithms, one subfolder per category.
#
# Globbed with CONFIGURE_DEPENDS so dropping a new .cpp into any category
# folder is picked up on the next build without editing this file — adding an
# algorithm should cost exactly one file.
#
# MUST be an OBJECT library. Algorithms self-register via a static initializer
# and nothing references their symbols, so a STATIC library would be stripped
# by the linker and they would silently vanish from the registry.
file(GLOB_RECURSE TGLAB_ALGORITHM_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_SOURCE_DIR}/src/algorithms/*.cpp)
add_library(tglab_algorithms OBJECT ${TGLAB_ALGORITHM_SOURCES})
target_link_libraries(tglab_algorithms PUBLIC tglab_core)

# ---------------------------------------------------------------------------
add_executable(tglab
    src/app/main.cpp
    src/app/image_view.cpp
    src/app/file_watch.cpp
    $<TARGET_OBJECTS:tglab_algorithms>
)
target_link_libraries(tglab PRIVATE tglab_core imgui d3d12 dxgi dbghelp)

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
# DXC redistributables.
#
# dxcompiler.dll needs dxil.dll beside it to sign shaders; without the signing
# step D3D12 refuses to load the resulting DXIL, with an error that does not
# mention the missing DLL. Copy both next to every executable that compiles
# shaders.
find_file(DXC_COMPILER_DLL dxcompiler.dll
    PATHS "$ENV{WindowsSdkDir}bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
          "C:/Program Files (x86)/Windows Kits/10/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
          "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
    NO_DEFAULT_PATH)
find_file(DXC_DXIL_DLL dxil.dll
    PATHS "$ENV{WindowsSdkDir}bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
          "C:/Program Files (x86)/Windows Kits/10/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
          "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
    NO_DEFAULT_PATH)

function(tglab_copy_dxc target)
    if(DXC_COMPILER_DLL AND DXC_DXIL_DLL)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DXC_COMPILER_DLL}" "$<TARGET_FILE_DIR:${target}>"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DXC_DXIL_DLL}" "$<TARGET_FILE_DIR:${target}>"
            VERBATIM)
    else()
        message(WARNING "dxcompiler.dll/dxil.dll not found — GPU shaders will not compile at runtime")
    endif()
endfunction()

tglab_copy_dxc(tglab)

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

# Runtime checks: worker thread + GPU compute. Needs a D3D12 device and takes
# seconds, so it is a separate binary from the fast script tests.
add_executable(tglab_runtime_tests
    tests/test_runtime.cpp
    $<TARGET_OBJECTS:tglab_algorithms>
)
target_link_libraries(tglab_runtime_tests PRIVATE tglab_core imgui d3d12 dxgi)
if(MSVC)
    target_compile_definitions(tglab_runtime_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
tglab_copy_dxc(tglab_runtime_tests)

# Behavioural checks for the filter family: noise reduction, edge preservation,
# and flat-field stability. No window and no GPU, so it runs with the spine.
add_executable(tglab_filter_tests
    tests/test_filters.cpp
    $<TARGET_OBJECTS:tglab_algorithms>
)
target_link_libraries(tglab_filter_tests PRIVATE tglab_core imgui d3d12 dxgi)
if(MSVC)
    target_compile_definitions(tglab_filter_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Demosaic behaviour: reconstruction against a known scene, CFA phase, and
# the metadata rules that keep a demosaiced image from being demosaiced twice.
add_executable(tglab_demosaic_tests
    tests/test_demosaic.cpp
    $<TARGET_OBJECTS:tglab_algorithms>
)
target_link_libraries(tglab_demosaic_tests PRIVATE tglab_core imgui d3d12 dxgi)
if(MSVC)
    target_compile_definitions(tglab_demosaic_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
tglab_copy_dxc(tglab_demosaic_tests)

enable_testing()
add_test(NAME script_spine COMMAND tglab_tests)
add_test(NAME filters      COMMAND tglab_filter_tests)
add_test(NAME demosaic     COMMAND tglab_demosaic_tests)
add_test(NAME runtime      COMMAND tglab_runtime_tests)
