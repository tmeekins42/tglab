# Writes the README that ships inside the binary archive.
#
# Generated rather than checked in, so the version it states cannot disagree
# with the build it accompanies. Run by the package_release target.
#
# Deliberately short: someone who has just unzipped a folder wants to know how
# to start it and what to do if it does not start. The full documentation is in
# README.md, which ships alongside.

file(WRITE "${DIST_DIR}/READ-ME-FIRST.txt"
"tglab ${TGLAB_VERSION}
A lab bench for computer vision and computer graphics algorithms.
https://github.com/tmeekins42/tglab


RUNNING IT
----------
Double-click tglab.exe. It opens scripts/hello.tgl and loads assets/test.png.

Drag image files onto the window to add them to the palette; raw files (CR2,
CR3, NEF, ARW, DNG and friends) are decoded with LibRaw. Edit a .tgl script in
any text editor and save -- the app reloads it automatically.

You can also pass a script and images on the command line:

    tglab.exe scripts\\thresholds.tgl myscan.png


REQUIREMENTS
------------
Windows 10 or 11, 64-bit, with a Direct3D 12 GPU.

If tglab.exe does not start and says something about VCRUNTIME140.dll or
MSVCP140.dll being missing, install the Microsoft Visual C++ Redistributable
for Visual Studio 2015-2022 (x64):

    https://aka.ms/vs/17/release/vc_redist.x64.exe

Keep dxcompiler.dll and dxil.dll next to tglab.exe. They compile the GPU
shaders at runtime; without them every algorithm silently falls back to the
CPU and large images become very slow.

Keep the scripts and assets folders next to tglab.exe too, or the app starts
with nothing loaded.


LICENCE
-------
tglab is MIT licensed -- see LICENSE.

It bundles third-party software whose licences are in the licenses folder, and
which are listed in the app under Help -> About tglab.

LibRaw is licensed under the LGPL v2.1 or the CDDL v1.0, at your option, and is
linked statically into tglab.exe. Its complete source is in the tglab
repository under third_party/libraw/, and building tglab from that repository
rebuilds LibRaw from that source -- so a modified LibRaw can be substituted by
editing it there and rebuilding.

LibRaw is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.
")
