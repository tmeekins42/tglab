# LibRaw as a static library.
#
# LibRaw ships autotools and MSVC makefiles but no CMakeLists, so this builds
# it from a source glob rather than vendoring a fork. The file list is stable
# across releases and the pinned tag makes it reproducible.
#
# The optional back-ends (RawSpeed, Adobe DNG SDK) compile to no-ops unless
# their SDKs are present, so they can be included unconditionally -- their glue
# files are guarded internally by USE_RAWSPEED / USE_DNGSDK, neither of which
# is defined here.

set(LIBRAW_DIR ${CMAKE_SOURCE_DIR}/third_party/libraw)

if(NOT EXISTS ${LIBRAW_DIR}/src)
    message(FATAL_ERROR
        "third_party/libraw is empty. Run: git submodule update --init")
endif()

file(GLOB_RECURSE LIBRAW_SOURCES CONFIGURE_DEPENDS ${LIBRAW_DIR}/src/*.cpp)

# The *_ph.cpp files are placeholder stubs for building LibRaw *without*
# postprocessing -- they are alternatives to the real implementations, not
# additions, and including both gives duplicate-symbol link errors. We want the
# real ones, since dcraw_process() is the whole point of this path.
list(FILTER LIBRAW_SOURCES EXCLUDE REGEX "_ph\\.cpp$")

add_library(libraw STATIC ${LIBRAW_SOURCES})
target_include_directories(libraw PUBLIC ${LIBRAW_DIR})

if(MSVC)
    # LibRaw is decades-old C-style code carried forward from dcraw. It compiles
    # with warnings that are noise here, and treating them as errors (or even
    # printing them) would bury warnings from tglab's own code.
    target_compile_options(libraw PRIVATE /w)
    # Its datastream layer uses fopen/sprintf and similar throughout.
    target_compile_definitions(libraw PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# LIBRAW_NODLL: build/link statically rather than importing from a DLL. Without
# it the headers declare the API dllimport and linking fails with unresolved
# externals that do not obviously point at this setting.
target_compile_definitions(libraw PUBLIC LIBRAW_NODLL)

# ws2_32 for ntohl/htonl, which LibRaw's decoders use for byte swapping.
if(WIN32)
    target_link_libraries(libraw PUBLIC ws2_32)
endif()
