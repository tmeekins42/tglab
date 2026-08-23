# Binary packaging: `cmake --build build --config Release --target package_release`
#
# Produces dist/tglab-<version>-win64.zip containing everything needed to run on
# a machine that has never seen the source. That is the whole point of this
# target -- "download, unzip, double-click" -- so the layout is flat and every
# runtime dependency travels with the exe.
#
# What ships and why:
#
#   tglab.exe             the app
#   dxcompiler.dll        DXC, from the SDK's Redist/D3D -- shaders compile at
#   dxil.dll              runtime, so these are not optional. dxil.dll signs the
#                         DXIL; without it D3D12 refuses the shader with an error
#                         that never mentions the missing DLL.
#   scripts/              the app opens scripts/hello.tgl at startup
#   assets/               and loads assets/test.png when given no image
#   LICENSE               tglab's own MIT terms
#   licenses/             third-party texts. LibRaw is LGPL-2.1/CDDL-1.0 and is
#                         linked statically, so shipping its notice and licence
#                         is an obligation, not a courtesy.
#   README.md             so the archive explains itself offline
#
# ResolveDataPath() searches beside the executable, so scripts/ and assets/ sit
# next to tglab.exe rather than needing an install step.

# Missing DLLs are a warning for an ordinary build -- the app still runs, just
# on the CPU -- but they are fatal for a release. An archive without them
# starts and then quietly falls back to the CPU on every algorithm, which is
# the worst outcome: slow, and with nothing to say why.
if(NOT DXC_COMPILER_DLL OR NOT DXC_DXIL_DLL)
    add_custom_target(package_release
        COMMAND ${CMAKE_COMMAND} -P "${CMAKE_SOURCE_DIR}/cmake/fail.cmake"
        VERBATIM)
    return()
endif()

set(TGLAB_DIST_NAME "tglab-${PROJECT_VERSION}-win64")
set(TGLAB_DIST_DIR  "${CMAKE_BINARY_DIR}/dist/${TGLAB_DIST_NAME}")

add_custom_target(package_release
    COMMENT "Packaging ${TGLAB_DIST_NAME}.zip"

    # Start clean: a stale file from a previous version would otherwise ship
    # silently, and a release archive is the worst place to find one.
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${TGLAB_DIST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TGLAB_DIST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TGLAB_DIST_DIR}/licenses"

    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:tglab>" "${TGLAB_DIST_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy "${DXC_COMPILER_DLL}"   "${TGLAB_DIST_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy "${DXC_DXIL_DLL}"       "${TGLAB_DIST_DIR}/"

    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/scripts" "${TGLAB_DIST_DIR}/scripts"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/assets"  "${TGLAB_DIST_DIR}/assets"

    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_SOURCE_DIR}/LICENSE"   "${TGLAB_DIST_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_SOURCE_DIR}/README.md" "${TGLAB_DIST_DIR}/"

    # Third-party licence texts, named so it is obvious which is which.
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_SOURCE_DIR}/third_party/imgui/LICENSE.txt"
            "${TGLAB_DIST_DIR}/licenses/imgui-LICENSE.txt"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_SOURCE_DIR}/third_party/libraw/LICENSE.LGPL"
            "${TGLAB_DIST_DIR}/licenses/libraw-LICENSE.LGPL"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_SOURCE_DIR}/third_party/libraw/LICENSE.CDDL"
            "${TGLAB_DIST_DIR}/licenses/libraw-LICENSE.CDDL"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_SOURCE_DIR}/third_party/libraw/COPYRIGHT"
            "${TGLAB_DIST_DIR}/licenses/libraw-COPYRIGHT"
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_SOURCE_DIR}/third_party/stb/LICENSE.txt"
            "${TGLAB_DIST_DIR}/licenses/stb-LICENSE.txt"

    # Generated rather than checked in, so it cannot disagree with the archive
    # it describes.
    COMMAND ${CMAKE_COMMAND}
            -DDIST_DIR=${TGLAB_DIST_DIR}
            -DTGLAB_VERSION=${PROJECT_VERSION}
            -P "${CMAKE_SOURCE_DIR}/cmake/write_readme.cmake"

    COMMAND ${CMAKE_COMMAND} -E chdir "${CMAKE_BINARY_DIR}/dist"
            ${CMAKE_COMMAND} -E tar cf "${TGLAB_DIST_NAME}.zip" --format=zip
            "${TGLAB_DIST_NAME}"

    COMMAND ${CMAKE_COMMAND} -E echo
            "packaged: ${CMAKE_BINARY_DIR}/dist/${TGLAB_DIST_NAME}.zip"
    VERBATIM)

# Depends on the app, so `--target package_release` builds it first rather than
# quietly zipping a stale exe.
add_dependencies(package_release tglab)
