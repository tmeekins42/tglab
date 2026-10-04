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

// HOW FRAMES ARE CHOSEN.
//
// BY MOTION, the default: a frame is kept each time the view has moved
// `step` of the frame's shorter side since the last one kept, measured by
// tracking corners from frame to frame, or when fewer than `minTracked` of
// them are still in view -- and at least every 1/`floorFrames` of the clip,
// however little it moved. A fast stretch gives many frames, so a long
// walk-around keeps its overlap wherever it speeds up, where equal time
// slots spread thinner the longer the clip. The frame kept is the sharpest
// since the view moved half a step (or half the floor's interval passed).
// `maxFrames` caps the total: past it every other frame goes and the step
// and the floor's interval double.
//
// THE CAP IS A MEMORY GUARD, NOT A QUALITY SETTING: every halving doubles
// the step, and the step is what decides whether a clip reconstructs. It
// was 250, and a 219 s walk around a room (IMG_1537) wanted 1577 frames at
// 4%: halved three times to a 32% step, neighbouring frames 22 degrees
// apart, and only 70 of 218 cameras reconstructed. Uncapped, all 1577 did.
// At 1080x1920 a kept frame is 8 MB of RAM, so 2000 is about 16 GB.
//
// BY TIME: `frames` equal time slots, each keeping its sharpest frame -- the
// original scheme, kept for comparison and for the tests of slotting.
//
// MEASURED on six phone clips, cameras reconstructed / frames kept:
//
//   clip      length   100 time slots   motion 5%   motion ~4%
//   IMG_1529     8 s     100/100          20/20       (face, close orbit)
//   IMG_1528    25 s     100/100         146/146
//   IMG_1534    39 s      45/100         179/179
//   IMG_1525    57 s      40/100         101/101
//   IMG_1526    59 s      34/100         107/141     178/178
//   IMG_1537   219 s                                1577/1577   (a room)
//
// Equal slots fail every clip over half a minute. The step is the coarsest
// that reconstructed all of them; the floor is for the short clips, which
// reconstruct at any step but keep far fewer points without it (the face:
// 3141 at 5%, against 23232 from 100 slots).
enum class VideoPick { Motion, Time };

struct VideoOptions {
    VideoPick pick = VideoPick::Motion;
    // By motion; see above.
    double step = 0.04;
    double minTracked = 0.6;
    int    floorFrames = 60;
    int    maxFrames = 2000;
    // A clip that moves too little for this many falls back to time slots.
    int    minFrames = 8;
    int frames = 100;     // by time: slots, so at most this many frames out
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
    // By motion: each kept frame's movement since the one before (a
    // fraction of the shorter side), why frames were kept, the step it ended
    // at, and how often the cap halved the set.
    std::vector<double> motion;
    int    byMotion = 0, byTracking = 0, byTime = 0, thinned = 0;
    double step = 0.0;
    bool   byTimeSlots = false;     // chosen by time, asked for or fallen back to
};

// A line for a status bar: how many frames came from how many, and how.
std::string VideoNote(const VideoInfo& info, int kept);

// Reads `path` and returns its chosen frames, in time order, as RGBA8.
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
