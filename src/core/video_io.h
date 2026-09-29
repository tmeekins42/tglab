// video_io — a video file in, a group of well-chosen frames out.
//
// WHY A VIDEO IS NOT JUST "EVERY FRAME". Thirty seconds of phone video is
// 900 frames, and for reconstruction almost all of them are redundant:
// neighbouring frames are a thirtieth of a second apart and see the scene
// from nearly the same place, which gives the SfM chain no parallax to work
// with and the memory budget nothing back. Some are also motion-blurred, and
// a blurred frame costs more than it gives -- features detect worse on it and
// splat training learns the blur.
//
// So the clip is divided into `frames` equal TIME SLOTS and each slot keeps
// its SHARPEST frame, scored by the variance of the Laplacian of its luma --
// the standard blur measure: sharp detail has strong second derivatives, and
// motion blur smooths them away. Even spacing and sharpness in one pass, and
// only one frame per slot is ever held in memory.
//
// DECODED WITH MEDIA FOUNDATION, Windows' own video stack, so there is no new
// dependency: MP4 and MOV with H.264 work on any Windows 10/11, and HEVC --
// what iPhones record by default -- once the free "HEVC Video Extensions" are
// installed from the Microsoft Store. Without them an HEVC clip fails to open
// and the error says so.
//
// A PHONE'S ORIENTATION is honoured. Portrait video is stored landscape with
// a rotation flag, and frames are rotated upright here: a reconstruction of
// a sideways scene works, but everything downstream -- the viewer's up, a
// human looking at the frames -- would have to cope with it.
#pragma once

#include <string>
#include <vector>

#include "image.h"

namespace tglab {

// True for the extensions this treats as video.
bool IsVideoPath(const std::string& path);

struct VideoOptions {
    int frames = 100;     // time slots, so at most this many frames out
    int maxDim = 1920;    // longer side after decoding; 0 keeps full size
};

struct VideoInfo {
    int    width = 0, height = 0;   // as decoded, before rotation/downscale
    int    rotation = 0;            // degrees clockwise applied: 0/90/180/270
    double seconds = 0.0;
    double fps = 0.0;
    int    decoded = 0;             // frames read from the file
    std::vector<double> times;      // each kept frame's time, seconds
    std::vector<double> sharpness;  // and its score
};

// Reads `path` and returns one frame per time slot, in time order, as RGBA8.
bool LoadVideoFrames(const std::string& path, const VideoOptions& opt,
                     std::vector<Image>* frames, VideoInfo* info, std::string* err);

// The blur score: variance of the 4-neighbour Laplacian of luma, sampled on
// a grid so a 4K frame costs about what a 480p one would. Exposed so the
// test can check it ranks sharp above blurred.
double FrameSharpness(const uint8_t* rgba, int w, int h, int pitch);

// The row pitch of a decoded w x h BGRX frame held in a plain buffer of
// `bytes`: the decoder pads either the rows (a portrait 1080-wide frame
// arrives with 1088-pixel rows) or the row count (1080 -> 1088), and only
// the length tells which.
int BufferPitch(uint32_t bytes, uint32_t w, uint32_t h);

// One decoded frame to an upright RGBA8 image: `bgrx` is `w` x `h` pixels
// at `pitch` bytes per row (4 bytes each, BGRX when `isBgr`, else RGBX),
// box-downscaled by `k`, then rotated `rotation` degrees clockwise (0, 90,
// 180, 270). Exposed so the rotation can be tested on known pixels.
Image FrameToImage(const uint8_t* bgrx, int w, int h, int pitch, int k, int rotation,
                   bool isBgr);

// Writes `frames` (RGBA8, all one size) as an H.264 MP4 at `fps`, with an
// optional rotation flag. For tests: a real clip made from known pixels, so
// decoding, slotting and rotation can be checked without shipping a video.
bool WriteTestVideo(const std::string& path, const std::vector<Image>& frames,
                    double fps, int rotation, std::string* err);

}  // namespace tglab
