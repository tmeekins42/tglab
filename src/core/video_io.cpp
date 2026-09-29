#include "video_io.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>

#include <windows.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <propvarutil.h>

namespace tglab {
namespace {

// COM and Media Foundation for the calling thread, for one call's duration.
// Both are reference counted, so nesting with anything else the thread does
// is safe. RPC_E_CHANGED_MODE means the thread already has COM in another
// apartment mode -- usable, but not ours to uninitialise.
struct MfScope {
    bool com = false, mf = false;
    MfScope() {
        const HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        com = SUCCEEDED(hr);
        mf = SUCCEEDED(MFStartup(MF_VERSION));
    }
    ~MfScope() {
        if (mf) MFShutdown();
        if (com) CoUninitialize();
    }
};

template <typename T>
void SafeRelease(T** p) {
    if (*p) { (*p)->Release(); *p = nullptr; }
}

std::wstring Widen(const std::string& s) {
    const int n = MultiByteToWideChar(CP_ACP, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(size_t(std::max(n, 1)), L'\0');
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), -1, w.data(), n);
    if (!w.empty() && w.back() == L'\0') w.pop_back();
    return w;
}

std::string HrText(HRESULT hr) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "0x%08lX", static_cast<unsigned long>(hr));
    return buf;
}

}  // namespace

bool IsVideoPath(const std::string& path) {
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string ext = path.substr(dot + 1);
    for (char& c : ext) c = char(std::tolower(static_cast<unsigned char>(c)));
    return ext == "mp4" || ext == "mov" || ext == "m4v" || ext == "avi" ||
           ext == "wmv" || ext == "mkv";
}

int BufferPitch(uint32_t bytes, uint32_t w, uint32_t h) {
    // The decoder pads to its macroblock grid, and a plain buffer does not
    // say which way: a 1920x1080 frame arrives with 1088 rows, a 1080x1920
    // one with 1088-pixel rows. MF_MT_DEFAULT_STRIDE reports the unpadded
    // width either way, so the layout is read from the length instead.
    const uint32_t tight = w * 4;
    if (h == 0 || bytes <= tight * h) return int(tight);
    const uint32_t rowsPadded = (h + 15) & ~15u;
    for (const uint32_t rows : {rowsPadded, h}) {
        if (bytes % rows) continue;
        const uint32_t pitch = bytes / rows;
        if (pitch >= tight && pitch - tight < 256 && pitch % 4 == 0) return int(pitch);
    }
    return int(tight);
}

double FrameSharpness(const uint8_t* px, int w, int h, int pitch) {
    // Sampled on a grid of about 360 rows, each sample a full-resolution
    // Laplacian: the grid keeps a 4K frame cheap, and the one-pixel
    // neighbourhood keeps it measuring the finest detail, which is the
    // detail motion blur removes first.
    const int step = std::max(1, std::min(w, h) / 360);
    // (r + 2g + b) / 4: symmetric in r and b, so BGRA and RGBA score alike.
    auto luma = [&](int x, int y) {
        const uint8_t* p = px + ptrdiff_t(y) * pitch + ptrdiff_t(x) * 4;
        return (double(p[0]) + 2.0 * double(p[1]) + double(p[2])) * 0.25;
    };
    double s = 0.0, s2 = 0.0;
    long long n = 0;
    for (int y = 1; y < h - 1; y += step)
        for (int x = 1; x < w - 1; x += step) {
            const double l = 4.0 * luma(x, y) - luma(x - 1, y) - luma(x + 1, y) -
                             luma(x, y - 1) - luma(x, y + 1);
            s += l;
            s2 += l * l;
            ++n;
        }
    if (n == 0) return 0.0;
    const double m = s / double(n);
    return s2 / double(n) - m * m;
}

// One decoded frame (BGRX rows, any pitch sign handled by the caller passing
// the first row's address) to an upright RGBA8 image: box-downscaled by `k`,
// then rotated `rotation` degrees clockwise.
Image FrameToImage(const uint8_t* bgrx, int w, int h, int pitch, int k, int rotation,
                   bool isBgr) {
    k = std::max(1, k);
    const int dw = std::max(1, w / k), dh = std::max(1, h / k);
    const bool swapWH = rotation == 90 || rotation == 270;
    const int ow = swapWH ? dh : dw, oh = swapWH ? dw : dh;

    Image img;
    img.Alloc(ImageDesc{ow, oh, Format::RGBA8});
    ImageView v = img.MapCpuWrite();
    const int ri = isBgr ? 2 : 0, bi = isBgr ? 0 : 2;
    const double inv = 1.0 / double(k * k);
    for (int oy = 0; oy < oh; ++oy)
        for (int ox = 0; ox < ow; ++ox) {
            // Where this output pixel comes from in the downscaled source.
            int sx = ox, sy = oy;
            switch (rotation) {
                case 90:  sx = oy;            sy = dh - 1 - ox; break;
                case 180: sx = dw - 1 - ox;   sy = dh - 1 - oy; break;
                case 270: sx = dw - 1 - oy;   sy = ox;          break;
                default: break;
            }
            double c[3] = {0, 0, 0};
            for (int yy = 0; yy < k; ++yy) {
                const uint8_t* row = bgrx + ptrdiff_t(sy * k + yy) * pitch;
                for (int xx = 0; xx < k; ++xx) {
                    const uint8_t* p = row + ptrdiff_t(sx * k + xx) * 4;
                    c[0] += p[ri];
                    c[1] += p[1];
                    c[2] += p[bi];
                }
            }
            uint8_t* o = v.At<uint8_t>(ox, oy);
            for (int ch = 0; ch < 3; ++ch) o[ch] = uint8_t(c[ch] * inv + 0.5);
            o[3] = 255;
        }
    return img;
}

bool LoadVideoFrames(const std::string& path, const VideoOptions& opt,
                     std::vector<Image>* frames, VideoInfo* info, std::string* err) {
    MfScope scope;
    if (!scope.mf) { *err = "Media Foundation is not available"; return false; }

    IMFAttributes* attr = nullptr;
    IMFSourceReader* reader = nullptr;
    IMFMediaType* native = nullptr;
    IMFMediaType* want = nullptr;
    IMFMediaType* cur = nullptr;
    auto cleanup = [&] {
        SafeRelease(&cur);
        SafeRelease(&want);
        SafeRelease(&native);
        SafeRelease(&reader);
        SafeRelease(&attr);
    };

    MFCreateAttributes(&attr, 1);
    // The reader's own converter, so any codec's output arrives as RGB32.
    if (attr) attr->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
    HRESULT hr = MFCreateSourceReaderFromURL(Widen(path).c_str(), attr, &reader);
    if (FAILED(hr)) {
        *err = "could not open '" + path + "' as a video (" + HrText(hr) + ")";
        cleanup();
        return false;
    }
    const DWORD kStream = DWORD(MF_SOURCE_READER_FIRST_VIDEO_STREAM);
    reader->SetStreamSelection(DWORD(MF_SOURCE_READER_ALL_STREAMS), FALSE);
    reader->SetStreamSelection(kStream, TRUE);

    // The rotation flag and codec live on the NATIVE type.
    GUID codec = GUID_NULL;
    UINT32 rot = 0;
    if (SUCCEEDED(reader->GetNativeMediaType(kStream, 0, &native))) {
        native->GetGUID(MF_MT_SUBTYPE, &codec);
        native->GetUINT32(MF_MT_VIDEO_ROTATION, &rot);
    }

    MFCreateMediaType(&want);
    want->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    want->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
    hr = reader->SetCurrentMediaType(kStream, nullptr, want);
    if (FAILED(hr)) {
        *err = "could not decode '" + path + "' (" + HrText(hr) + ")";
        if (codec == MFVideoFormat_HEVC)
            *err += ": it is HEVC, which Windows decodes only with the \"HEVC "
                    "Video Extensions\" from the Microsoft Store -- or record "
                    "H.264 (iPhone: Settings > Camera > Formats > Most Compatible)";
        cleanup();
        return false;
    }
    reader->GetCurrentMediaType(kStream, &cur);
    UINT32 w = 0, h = 0, num = 0, den = 0;
    MFGetAttributeSize(cur, MF_MT_FRAME_SIZE, &w, &h);
    MFGetAttributeRatio(cur, MF_MT_FRAME_RATE, &num, &den);
    if (w == 0 || h == 0) {
        *err = "'" + path + "' reports no frame size";
        cleanup();
        return false;
    }

    PROPVARIANT var;
    PropVariantInit(&var);
    LONGLONG duration = 0;   // 100 ns units
    if (SUCCEEDED(reader->GetPresentationAttribute(DWORD(MF_SOURCE_READER_MEDIASOURCE),
                                                   MF_PD_DURATION, &var)))
        duration = LONGLONG(var.uhVal.QuadPart);
    PropVariantClear(&var);

    info->width = int(w);
    info->height = int(h);
    info->rotation = int(rot) % 360;
    info->seconds = double(duration) * 1e-7;
    info->fps = den ? double(num) / double(den) : 0.0;
    info->decoded = 0;
    info->times.clear();
    info->sharpness.clear();

    const int slots = std::max(1, opt.frames);
    const int longSide = int(std::max(w, h));
    const int k = (opt.maxDim > 0 && longSide > opt.maxDim)
                      ? (longSide + opt.maxDim - 1) / opt.maxDim : 1;

    // The best candidate of the slot being filled, held raw and converted
    // only once the slot closes: each improvement is then a copy, not a
    // downscale and rotation.
    std::vector<uint8_t> cand;
    double candScore = -1.0, candTime = 0.0;
    int candSlot = -1;
    frames->clear();
    auto flush = [&] {
        if (candSlot < 0) return;
        frames->push_back(FrameToImage(cand.data(), int(w), int(h), int(w) * 4, k,
                                       info->rotation, true));
        info->times.push_back(candTime);
        info->sharpness.push_back(candScore);
        candSlot = -1;
        candScore = -1.0;
    };

    for (;;) {
        DWORD flags = 0;
        LONGLONG ts = 0;
        IMFSample* sample = nullptr;
        hr = reader->ReadSample(kStream, 0, nullptr, &flags, &ts, &sample);
        if (FAILED(hr)) {
            SafeRelease(&sample);
            *err = "decoding '" + path + "' failed after " +
                   std::to_string(info->decoded) + " frames (" + HrText(hr) + ")";
            cleanup();
            return false;
        }
        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) { SafeRelease(&sample); break; }
        if (!sample) continue;

        IMFMediaBuffer* buf = nullptr;
        if (SUCCEEDED(sample->ConvertToContiguousBuffer(&buf))) {
            // Lock2D gives the true first row and pitch, whose SIGN says
            // whether the frame is stored bottom-up -- RGB32 often is.
            IMF2DBuffer* b2 = nullptr;
            BYTE* scan0 = nullptr;
            LONG pitch = 0;
            BYTE* raw = nullptr;
            DWORD len = 0;
            bool locked2d = false, locked = false;
            if (SUCCEEDED(buf->QueryInterface(IID_PPV_ARGS(&b2))) &&
                SUCCEEDED(b2->Lock2D(&scan0, &pitch))) {
                locked2d = true;
            } else if (SUCCEEDED(buf->Lock(&raw, nullptr, &len))) {
                locked = true;
                scan0 = raw;
                pitch = BufferPitch(len, w, h);
                if (len < DWORD(pitch) * h) scan0 = nullptr;   // truncated frame
            }
            if (scan0) {
                const double t = double(ts) * 1e-7;
                const int slot = duration > 0
                    ? std::min(slots - 1, int(double(ts) / double(duration) * slots))
                    : info->decoded;
                if (slot != candSlot && candSlot >= 0) flush();
                if (duration > 0 || slot < slots) {
                    const double score = FrameSharpness(scan0, int(w), int(h), int(pitch));
                    if (score > candScore) {
                        // Copied top-down, whatever the stored order.
                        cand.resize(size_t(w) * size_t(h) * 4);
                        for (UINT32 y = 0; y < h; ++y)
                            std::memcpy(cand.data() + size_t(y) * w * 4,
                                        scan0 + ptrdiff_t(y) * pitch, size_t(w) * 4);
                        candScore = score;
                        candTime = t;
                        candSlot = slot;
                    }
                }
                ++info->decoded;
            }
            if (locked2d) b2->Unlock2D();
            if (locked) buf->Unlock();
            SafeRelease(&b2);
        }
        SafeRelease(&buf);
        SafeRelease(&sample);
    }
    flush();
    cleanup();

    if (frames->empty()) {
        *err = "'" + path + "' decoded no frames";
        return false;
    }
    return true;
}

bool WriteTestVideo(const std::string& path, const std::vector<Image>& frames,
                    double fps, int rotation, std::string* err) {
    if (frames.empty()) { *err = "no frames"; return false; }
    MfScope scope;
    if (!scope.mf) { *err = "Media Foundation is not available"; return false; }

    const ImageDesc d = frames[0].Desc();
    const UINT32 w = UINT32(d.width), h = UINT32(d.height);
    const UINT32 rate = UINT32(std::lround(fps));
    const LONGLONG dur = LONGLONG(1e7 / fps);

    IMFSinkWriter* writer = nullptr;
    IMFMediaType* out = nullptr;
    IMFMediaType* in = nullptr;
    auto cleanup = [&] {
        SafeRelease(&in);
        SafeRelease(&out);
        SafeRelease(&writer);
    };

    HRESULT hr = MFCreateSinkWriterFromURL(Widen(path).c_str(), nullptr, nullptr, &writer);
    DWORD stream = 0;
    if (SUCCEEDED(hr)) hr = MFCreateMediaType(&out);
    if (SUCCEEDED(hr)) {
        out->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        out->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264);
        out->SetUINT32(MF_MT_AVG_BITRATE, 8000000);
        out->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
        MFSetAttributeSize(out, MF_MT_FRAME_SIZE, w, h);
        MFSetAttributeRatio(out, MF_MT_FRAME_RATE, rate, 1);
        MFSetAttributeRatio(out, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
        if (rotation) out->SetUINT32(MF_MT_VIDEO_ROTATION, UINT32(rotation));
        hr = writer->AddStream(out, &stream);
    }
    if (SUCCEEDED(hr)) hr = MFCreateMediaType(&in);
    if (SUCCEEDED(hr)) {
        in->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        in->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
        in->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
        // Top-down, stated rather than left to the RGB default of bottom-up.
        in->SetUINT32(MF_MT_DEFAULT_STRIDE, UINT32(w * 4));
        MFSetAttributeSize(in, MF_MT_FRAME_SIZE, w, h);
        MFSetAttributeRatio(in, MF_MT_FRAME_RATE, rate, 1);
        MFSetAttributeRatio(in, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
        hr = writer->SetInputMediaType(stream, in, nullptr);
    }
    if (SUCCEEDED(hr)) hr = writer->BeginWriting();

    for (size_t i = 0; SUCCEEDED(hr) && i < frames.size(); ++i) {
        IMFMediaBuffer* buf = nullptr;
        IMFSample* s = nullptr;
        hr = MFCreateMemoryBuffer(w * h * 4, &buf);
        BYTE* p = nullptr;
        if (SUCCEEDED(hr)) hr = buf->Lock(&p, nullptr, nullptr);
        if (SUCCEEDED(hr)) {
            ImageView v = const_cast<Image&>(frames[i]).MapCpuRead();
            for (UINT32 y = 0; y < h; ++y)
                for (UINT32 x = 0; x < w; ++x) {
                    const uint8_t* src = v.At<uint8_t>(int(x), int(y));
                    BYTE* o = p + (size_t(y) * w + x) * 4;
                    o[0] = src[2]; o[1] = src[1]; o[2] = src[0]; o[3] = 255;
                }
            buf->Unlock();
            buf->SetCurrentLength(w * h * 4);
            hr = MFCreateSample(&s);
        }
        if (SUCCEEDED(hr)) hr = s->AddBuffer(buf);
        if (SUCCEEDED(hr)) hr = s->SetSampleTime(LONGLONG(i) * dur);
        if (SUCCEEDED(hr)) hr = s->SetSampleDuration(dur);
        if (SUCCEEDED(hr)) hr = writer->WriteSample(stream, s);
        SafeRelease(&s);
        SafeRelease(&buf);
    }
    if (SUCCEEDED(hr)) hr = writer->Finalize();
    cleanup();
    if (FAILED(hr)) {
        *err = "writing '" + path + "' failed (" + HrText(hr) + ")";
        return false;
    }
    return true;
}

}  // namespace tglab
