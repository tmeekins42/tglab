cmake_minimum_required(VERSION 3.20)
project(tglab VERSION 0.7.0 LANGUAGES CXX)

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
    src/algo_util/auto_develop.cpp
    src/algo_util/histogram.cpp
    src/algo_util/pixel_buffer.cpp
    src/algo_util/white_balance.cpp
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
    src/app/about.cpp
    src/app/image_view.cpp
    src/app/file_watch.cpp
    $<TARGET_OBJECTS:tglab_algorithms>
)
target_link_libraries(tglab PRIVATE tglab_core imgui d3d12 dxgi dbghelp)

# The About dialog reports this; a hardcoded string in the source would drift.
target_compile_definitions(tglab PRIVATE TGLAB_VERSION="${PROJECT_VERSION}")

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
#
# Redist/D3D is searched FIRST and matters for shipping: those are the copies
# Microsoft designates as redistributable, and they are not byte-identical to
# the ones under bin/. Developing against bin/ and shipping Redist/ would mean
# releasing something never actually run, so prefer Redist everywhere and let
# bin/ be the fallback for an SDK install that lacks it.
find_file(DXC_COMPILER_DLL dxcompiler.dll
    PATHS "$ENV{WindowsSdkDir}Redist/D3D/x64"
          "C:/Program Files (x86)/Windows Kits/10/Redist/D3D/x64"
          "$ENV{WindowsSdkDir}bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
          "C:/Program Files (x86)/Windows Kits/10/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
          "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
    NO_DEFAULT_PATH)
find_file(DXC_DXIL_DLL dxil.dll
    PATHS "$ENV{WindowsSdkDir}Redist/D3D/x64"
          "C:/Program Files (x86)/Windows Kits/10/Redist/D3D/x64"
          "$ENV{WindowsSdkDir}bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
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
add_test(NAME denoise      COMMAND tglab_denoise_tests)

# Manual reproduction harness, not an add_test().
#
# It drives the real App through real frames, which is the only way to reach
# faults that need genuine present/worker interleaving. Deliberately NOT a
# gating test: the palette-reallocation crash it reproduces
# (TGLAB_PALETTE_TEST=1) fires about one run in four, because it depends on a
# freed descriptor being reused before the next draw. A test that passes 75% of
# the time on broken code is worse than none -- it reads as a guarantee and is
# not one.
#
#   build/Debug/dbg_hang.exe scripts/hello.tgl assets/test.png
#   TGLAB_PALETTE_TEST=1 build/Debug/dbg_hang.exe scripts/hello.tgl assets/test.png
add_executable(dbg_hang
    tests/dbg_hang.cpp
    src/app/about.cpp
    src/app/image_view.cpp
    src/app/file_watch.cpp
    $<TARGET_OBJECTS:tglab_algorithms>
)
target_link_libraries(dbg_hang PRIVATE tglab_core imgui d3d12 dxgi dbghelp)
target_compile_definitions(dbg_hang PRIVATE TGLAB_VERSION="${PROJECT_VERSION}")
if(MSVC)
    target_compile_definitions(dbg_hang PRIVATE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# GPU shader audit: runs every GPU-capable algorithm under the debug layer and
# GPU-based validation. Not an add_test() -- GBV is 10-100x slower and the run
# takes minutes, so it is a tool you invoke deliberately rather than part of
# the normal suite.
add_executable(gpu_audit
    tests/gpu_audit.cpp
    $<TARGET_OBJECTS:tglab_algorithms>
)
target_link_libraries(gpu_audit PRIVATE tglab_core d3d12 dxgi)
if(MSVC)
    target_compile_definitions(gpu_audit PRIVATE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# VRAM leak check: re-runs a pipeline many times and reports whether video
# memory is still climbing after warm-up. Not an add_test() -- it measures a
# shared adapter, so another process allocating during the run can skew it.
add_executable(gpu_leak
    tests/gpu_leak.cpp
    $<TARGET_OBJECTS:tglab_algorithms>
)
target_link_libraries(gpu_leak PRIVATE tglab_core d3d12 dxgi)
if(MSVC)
    target_compile_definitions(gpu_leak PRIVATE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Binary release packaging (see cmake/package.cmake).
include(${CMAKE_SOURCE_DIR}/cmake/package.cmake)

# Proves the packaged DXC actually compiles, not merely that the DLLs exist.
add_executable(dxc_smoke tests/dxc_smoke.cpp)
target_link_libraries(dxc_smoke PRIVATE tglab_core)
if(MSVC)
    target_compile_definitions(dxc_smoke PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
tglab_copy_dxc(dxc_smoke)
add_test(NAME dxc_smoke COMMAND dxc_smoke)

add_executable(tglab_denoise_tests tests/test_denoise.cpp $<TARGET_OBJECTS:tglab_algorithms>)
target_link_libraries(tglab_denoise_tests PRIVATE tglab_core)
if(MSVC)
    target_compile_definitions(tglab_denoise_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()








# Auto-develop: the measurement itself, and that its suggestions reach the
# declared controls as DEFAULTS rather than as values.
#
# Needs a raw file, which the repository does not carry, so it is not an
# add_test() -- point it at one:
#   build/Release/tglab_autodev_tests.exe <some.arw>
add_executable(tglab_autodev_tests
    tests/test_autodev.cpp
    $<TARGET_OBJECTS:tglab_algorithms>
)
target_link_libraries(tglab_autodev_tests PRIVATE tglab_core)
if(MSVC)
    target_compile_definitions(tglab_autodev_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# The base tone curve, and extracting the camera's embedded preview.
#
# The curve's properties need no photograph, so this one DOES run in ctest.
# Passing a raw file additionally checks preview extraction and what the
# measurement suggests for a real scene:
#   build/Release/tglab_tone_tests.exe <some.cr3>
add_executable(tglab_tone_tests
    tests/test_tone.cpp
    $<TARGET_OBJECTS:tglab_algorithms>
)
target_link_libraries(tglab_tone_tests PRIVATE tglab_core)
if(MSVC)
    target_compile_definitions(tglab_tone_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
add_test(NAME tone COMMAND tglab_tone_tests)







# ---------------------------------------------------------------------------
# Research tools. Not part of the app, not run by ctest -- these answer design
# questions and are kept because the answers are worth reproducing.
#
# headroom: how much accuracy is left above the current best demosaic, measured
# on real photographs at high-frequency sites. Written to decide whether a
# learned demosaic is worth building:
#   build/Release/headroom.exe <raw> [raw...]
add_executable(headroom tools/headroom.cpp $<TARGET_OBJECTS:tglab_algorithms>)
target_link_libraries(headroom PRIVATE tglab_core)
if(MSVC)
    target_compile_definitions(headroom PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# paper_data: regenerates the tables and figure crops under papers/, so every
# number in the write-up has reproducible provenance.
#   build/Release/paper_data.exe --table <raw>...
#   build/Release/paper_data.exe --crops <raw> <x> <y> <size> [outdir] [stops]
add_executable(paper_data tools/paper_data.cpp $<TARGET_OBJECTS:tglab_algorithms>)
target_link_libraries(paper_data PRIVATE tglab_core)
if(MSVC)
    target_compile_definitions(paper_data PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# overview: a whole frame rendered small with a coordinate grid, for picking
# crop regions to compare.
#   build/Release/overview.exe <raw> out.png [stops]
add_executable(overview tools/overview.cpp $<TARGET_OBJECTS:tglab_algorithms>)
target_link_libraries(overview PRIVATE tglab_core)
if(MSVC)
    target_compile_definitions(overview PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# VRAM growth when SWITCHING scripts, which gpu_leak does not cover -- that
# re-runs one pipeline and measures steady state. Each script builds a new
# pipeline holding kernels and scratch planes, so a missing release shows here.
#   build/Release/gpu_leak_scripts.exe [cycles] [image]
add_executable(gpu_leak_scripts tests/gpu_leak_scripts.cpp $<TARGET_OBJECTS:tglab_algorithms>)
target_link_libraries(gpu_leak_scripts PRIVATE tglab_core d3d12 dxgi)
if(MSVC)
    target_compile_definitions(gpu_leak_scripts PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
tglab_copy_dxc(gpu_leak_scripts)
