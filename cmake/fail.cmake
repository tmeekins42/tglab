# Fails the build, deliberately.
#
# `cmake -E false` exits 0 -- it is not one of the -E subcommands -- so a guard
# built on it silently succeeds, which is precisely the bug it was meant to
# prevent. message(FATAL_ERROR) actually fails.
message(FATAL_ERROR
    "Refusing to package without dxcompiler.dll and dxil.dll: the archive "
    "would start and then fall back to the CPU for every algorithm, with "
    "nothing to say why. Install the Windows SDK, or set DXC_COMPILER_DLL and "
    "DXC_DXIL_DLL explicitly.")
