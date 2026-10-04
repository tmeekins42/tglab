#include "video_io.h"
#include "parallel.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

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

// --- motion: a small sparse tracker ---------------------------------------------
//
// Only as much as picking frames needs: how far the view has moved since the
// last kept frame, and how much of it is still the same view. Grey frames a
// couple of hundred pixels across, Shi-Tomasi corners, pyramidal
// Lucas-Kanade -- the classic KLT tracker, a millisecond or so a frame.

struct Grey {
    int w = 0, h = 0;
    std::vector<float> px;
    float At(int x, int y) const { return px[size_t(y) * size_t(w) + size_t(x)]; }
    // Bilinear, clamped to the edge.
    float Sample(float x, float y) const {
        x = std::clamp(x, 0.0f, float(w - 1));
        y = std::clamp(y, 0.0f, float(h - 1));
        const int x0 = std::min(int(x), w - 2 < 0 ? 0 : w - 2);
        const int y0 = std::min(int(y), h - 2 < 0 ? 0 : h - 2);
        const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
        const float fx = x - float(x0), fy = y - float(y0);
        return (1 - fy) * ((1 - fx) * At(x0, y0) + fx * At(x1, y0)) +
               fy * ((1 - fx) * At(x0, y1) + fx * At(x1, y1));
    }
};

// Luma 0..1 of a BGRX/RGBX frame, box-downscaled by k: (r + 2g + b) / 4,
// symmetric in r and b so the channel order does not matter.
Grey GreyOf(const uint8_t* px, int w, int h, int pitch, int k) {
    Grey g;
    g.w = std::max(1, w / k);
    g.h = std::max(1, h / k);
    g.px.assign(size_t(g.w) * size_t(g.h), 0.0f);
    const float inv = 1.0f / (1020.0f * float(k * k));
    for (int y = 0; y < g.h; ++y)
        for (int x = 0; x < g.w; ++x) {
            int s = 0;
            for (int yy = 0; yy < k; ++yy) {
                const uint8_t* row = px + ptrdiff_t(y * k + yy) * pitch + ptrdiff_t(x * k) * 4;
                for (int xx = 0; xx < k; ++xx, row += 4) s += row[0] + 2 * row[1] + row[2];
            }
            g.px[size_t(y) * size_t(g.w) + size_t(x)] = float(s) * inv;
        }
    return g;
}

// Three levels, each with its central-difference gradients.
struct Pyramid {
    Grey lv[3], gx[3], gy[3];
    int levels = 0;
};

Pyramid PyramidOf(Grey base) {
    Pyramid p;
    p.lv[0] = std::move(base);
    p.levels = 1;
    for (int l = 1; l < 3; ++l) {
        const Grey& a = p.lv[l - 1];
        if (a.w < 16 || a.h < 16) break;
        Grey b;
        b.w = a.w / 2;
        b.h = a.h / 2;
        b.px.resize(size_t(b.w) * size_t(b.h));
        for (int y = 0; y < b.h; ++y)
            for (int x = 0; x < b.w; ++x)
                b.px[size_t(y) * size_t(b.w) + size_t(x)] =
                    0.25f * (a.At(2 * x, 2 * y) + a.At(2 * x + 1, 2 * y) +
                             a.At(2 * x, 2 * y + 1) + a.At(2 * x + 1, 2 * y + 1));
        p.lv[l] = std::move(b);
        p.levels = l + 1;
    }
    for (int l = 0; l < p.levels; ++l) {
        const Grey& a = p.lv[l];
        Grey& gx = p.gx[l];
        Grey& gy = p.gy[l];
        gx.w = gy.w = a.w;
        gx.h = gy.h = a.h;
        gx.px.assign(a.px.size(), 0.0f);
        gy.px.assign(a.px.size(), 0.0f);
        for (int y = 1; y + 1 < a.h; ++y)
            for (int x = 1; x + 1 < a.w; ++x) {
                gx.px[size_t(y) * size_t(a.w) + size_t(x)] = 0.5f * (a.At(x + 1, y) - a.At(x - 1, y));
                gy.px[size_t(y) * size_t(a.w) + size_t(x)] = 0.5f * (a.At(x, y + 1) - a.At(x, y - 1));
            }
    }
    return p;
}

struct Pt { float x = 0, y = 0; };

// Shi-Tomasi corners: the structure tensor's smaller eigenvalue over a 7x7
// window, the best one per cell of a 16 x 12 grid so they spread over the
// frame -- a view "still seen" should mean all of it, not one textured
// corner. Weak ones (under 5% of the strongest) are dropped: a blank wall
// gives none, and the caller notices.
std::vector<Pt> Corners(const Pyramid& p) {
    const Grey& gx = p.gx[0];
    const Grey& gy = p.gy[0];
    const int w = gx.w, h = gx.h, r = 3, margin = 8;
    const int cx = 16, cy = 12;
    std::vector<float> best(size_t(cx * cy), 0.0f);
    std::vector<Pt> at(size_t(cx * cy));
    float top = 0.0f;
    for (int y = margin; y < h - margin; y += 2)
        for (int x = margin; x < w - margin; x += 2) {
            float a = 0, b = 0, c = 0;
            for (int j = -r; j <= r; ++j)
                for (int i = -r; i <= r; ++i) {
                    const float ix = gx.At(x + i, y + j), iy = gy.At(x + i, y + j);
                    a += ix * ix;
                    b += ix * iy;
                    c += iy * iy;
                }
            const float mid = 0.5f * (a + c);
            const float lmin = mid - std::sqrt(std::max(0.0f, mid * mid - (a * c - b * b)));
            const size_t cell = size_t(std::min(cy - 1, y * cy / h) * cx + std::min(cx - 1, x * cx / w));
            if (lmin > best[cell]) { best[cell] = lmin; at[cell] = Pt{float(x), float(y)}; }
            top = std::max(top, lmin);
        }
    std::vector<Pt> out;
    for (size_t i = 0; i < best.size(); ++i)
        if (best[i] > 0.05f * top && best[i] > 1e-5f) out.push_back(at[i]);
    return out;
}

// Lucas-Kanade from A at `a` into B, coarse to fine; `b` is the guess in and
// the answer out. False when the window has no texture to lock onto or the
// point leaves the frame.
bool TrackPoint(const Pyramid& A, const Pyramid& B, Pt a, Pt* b) {
    const int r = 4;
    const int levels = std::min(A.levels, B.levels);
    float dx = (b->x - a.x) / float(1 << (levels - 1));
    float dy = (b->y - a.y) / float(1 << (levels - 1));
    for (int l = levels - 1; l >= 0; --l) {
        const float s = 1.0f / float(1 << l);
        const float ax = a.x * s, ay = a.y * s;
        const Grey& IA = A.lv[l];
        const Grey& IX = A.gx[l];
        const Grey& IY = A.gy[l];
        const Grey& IB = B.lv[l];
        float g11 = 0, g12 = 0, g22 = 0;
        float va[81], vx[81], vy[81];
        int n = 0;
        for (int j = -r; j <= r; ++j)
            for (int i = -r; i <= r; ++i, ++n) {
                va[n] = IA.Sample(ax + float(i), ay + float(j));
                vx[n] = IX.Sample(ax + float(i), ay + float(j));
                vy[n] = IY.Sample(ax + float(i), ay + float(j));
                g11 += vx[n] * vx[n];
                g12 += vx[n] * vy[n];
                g22 += vy[n] * vy[n];
            }
        const float det = g11 * g22 - g12 * g12;
        if (det < 1e-9f) return false;
        for (int it = 0; it < 12; ++it) {
            float b1 = 0, b2 = 0;
            n = 0;
            for (int j = -r; j <= r; ++j)
                for (int i = -r; i <= r; ++i, ++n) {
                    const float e = va[n] - IB.Sample(ax + dx + float(i), ay + dy + float(j));
                    b1 += e * vx[n];
                    b2 += e * vy[n];
                }
            const float ux = (g22 * b1 - g12 * b2) / det;
            const float uy = (g11 * b2 - g12 * b1) / det;
            dx += ux;
            dy += uy;
            if (ux * ux + uy * uy < 1e-4f) break;
        }
        if (l > 0) { dx *= 2.0f; dy *= 2.0f; }
    }
    b->x = a.x + dx;
    b->y = a.y + dy;
    const float w = float(A.lv[0].w), h = float(A.lv[0].h);
    return b->x >= 0 && b->y >= 0 && b->x < w && b->y < h;
}

// Tracked forward and back: kept only if it comes home to within a pixel.
// A point that drifted onto something else rarely finds its way back, so
// this is what makes "still tracked" mean the same view.
bool TrackChecked(const Pyramid& A, const Pyramid& B, Pt a, Pt* b) {
    if (!TrackPoint(A, B, a, b)) return false;
    Pt back = a;
    if (!TrackPoint(B, A, *b, &back)) return false;
    const float ex = back.x - a.x, ey = back.y - a.y;
    return ex * ex + ey * ey < 1.0f;
}

// --- picking frames by motion -----------------------------------------------------
//
// Points found in the last KEPT frame are followed frame to frame. When their
// median displacement reaches `step` (a fraction of the frame's shorter
// side), or too few are still tracked, a frame is kept -- not the frame that
// crossed the line but the SHARPEST since the view moved half a step, since
// motion blur arrives exactly when the camera speeds up. Tracking then
// restarts from the frame kept, so no two kept frames are more than a step
// apart, however the camera moved between them.
class MotionPicker {
public:
    MotionPicker(const VideoOptions& o, int w, int h, int rotation, double seconds,
                 std::vector<Image>* frames, VideoInfo* info)
        : m_o(o), m_w(w), m_h(h), m_rot(rotation), m_frames(frames), m_info(info) {
        const int longSide = std::max(w, h);
        m_k = (o.maxDim > 0 && longSide > o.maxDim) ? (longSide + o.maxDim - 1) / o.maxDim : 1;
        m_kt = std::max(1, (longSide + 239) / 240);   // tracking: ~240 px across
        m_step = o.step;
        // The floor in time: at least `floorFrames` over the clip.
        m_every = (o.floorFrames > 0 && seconds > 0.0) ? seconds / double(o.floorFrames) : 0.0;
    }

    // The tracking pyramid of a frame: a function of its pixels alone, so
    // any number of frames can have theirs built at once.
    Pyramid Prepare(const uint8_t* px, int pitch) const {
        return PyramidOf(GreyOf(px, m_w, m_h, pitch, m_kt));
    }

    void Frame(const uint8_t* px, int pitch, double t, double score) {
        Frame(px, pitch, t, score, Prepare(px, pitch));
    }

    // ...and the rest, which follows points from frame to frame and so must
    // see the frames one at a time, in order.
    void Frame(const uint8_t* px, int pitch, double t, double score, Pyramid p) {
        if (!m_started) {
            m_started = true;
            m_first = true;
            Seed(p, p);
            m_refTime = t;
            Offer(px, pitch, t, score, p, 0.0);
            m_prev = std::move(p);
            return;
        }
        int alive = 0;
        std::vector<double> dist;
        for (size_t i = 0; i < m_cur.size(); ++i) {
            if (!m_alive[i]) continue;
            Pt b = m_cur[i];
            if (TrackChecked(m_prev, p, m_cur[i], &b)) {
                m_cur[i] = b;
                ++alive;
                dist.push_back(std::hypot(double(b.x - m_ref[i].x), double(b.y - m_ref[i].y)));
            } else {
                m_alive[i] = 0;
            }
        }
        const double shortSide = double(std::min(p.lv[0].w, p.lv[0].h));
        double d = 0.0;
        if (!dist.empty()) {
            std::nth_element(dist.begin(), dist.begin() + long(dist.size() / 2), dist.end());
            d = dist[dist.size() / 2] / shortSide;
        }
        const double kept = m_seeded ? double(alive) / double(m_seeded) : 0.0;
        // Nothing to track -- a blank wall, darkness -- so time decides: the
        // sharpest frame of each second.
        const bool blind = m_seeded < 12;
        // Candidates lie between half a step and a step. The frame that
        // crosses the step is past it by up to a frame's motion, so it is
        // kept only when nothing inside qualified -- which is what makes a
        // step the most two kept frames can be apart.
        // THE FLOOR IN TIME works the same way: candidates from half the
        // interval, a frame kept by the whole of it. A slow or short clip
        // keeps its density this way; motion adds frames where it moves.
        const double lo = m_first ? 0.0 : 0.5 * m_step;
        const double hi = m_first ? 0.5 * m_step : m_step;
        const double every = blind ? 1.0 : m_every;
        const double since = t - m_refTime;

        // UNLESS THE WINDOW IS BLURRED. A fast stretch of the clip is a
        // blurred one, and there the window can hold only a frame or two,
        // all soft: kept anyway, they starve the matcher. On IMG_1535 a
        // second of turn kept frames of sharpness 22-37 where time slots
        // found 58-98 nearby; fewer than five tracks crossed it, and the walk
        // came out as two copies of the subject. So when the best candidate
        // is blurred -- under kSharpFrac of the clip's sharpness over the
        // last second -- the search runs on past the step, up to two steps,
        // and keeps the first sharp frame (or, failing one, the sharpest).
        // Sharp windows are unaffected: a step remains the most two kept
        // frames are apart wherever the clip is in focus.
        m_recent.push_back(score);
        if (m_recent.size() > 31) m_recent.erase(m_recent.begin());
        std::vector<double> rs = m_recent;
        std::nth_element(rs.begin(), rs.begin() + long(rs.size() / 2), rs.end());
        const double sharpEnough = kSharpFrac * rs[rs.size() / 2];
        auto candSharp = [&] { return m_haveCand && m_candScore >= sharpEnough; };

        // Nothing past either limit is a candidate -- the frame that crosses
        // one is kept only when nothing inside qualified -- unless the
        // window is blurred, as above.
        const bool reach = !candSharp();   // may look past the limits
        // Lost views -- a fast turn, something passing in front, a cut, and
        // blur itself, which defeats the tracker. Not faster than five a
        // second, or a stretch where nothing tracks would keep every frame.
        const bool lost = !blind && kept < m_o.minTracked && since >= 0.2;
        const bool inStep = d <= hi || (reach && d <= 2.0 * hi);
        const bool inTime = every <= 0.0 || since <= every || (reach && since <= 2.0 * every);
        const bool timeWindow = every > 0.0 && since >= 0.5 * every;
        // Lost and the best so far blurred: the measured motion means
        // little, so any frame is a candidate until a sharp one turns up.
        if ((inStep && inTime && (blind || d >= lo || timeWindow)) || (lost && reach))
            Offer(px, pitch, t, score, p, d);
        int why = 0;   // 1 motion, 2 tracking lost, 3 time
        if (blind) {
            if (since >= every) why = 3;
        } else if (d >= hi && (candSharp() || d >= 2.0 * hi)) {
            why = 1;
        } else if (lost && (candSharp() || since >= 1.0)) {
            // ...and if every frame since is blurred, the sharpest of a
            // second -- a fast turn's blur lasts less -- rather than waiting on.
            why = 2;
        } else if (m_every > 0.0 && since >= m_every &&
                   (candSharp() || since >= 2.0 * m_every)) {
            why = 3;
        }
        if (why) {
            if (!m_haveCand) Offer(px, pitch, t, score, p, d);
            Emit(p, why);
        }
        m_prev = std::move(p);
    }

    // The tail: a view half a step past the last kept frame is kept too, and
    // a clip too short to trigger anything still gives its sharpest frame.
    void Finish() {
        if (m_haveCand && (m_pending.empty() || m_candD >= 0.5 * m_step || m_first))
            Emit(m_prev, 1);
        m_info->step = m_step;
        m_frames->resize(m_pending.size());
        ParallelFor(m_pending.size(), [&](size_t i) {
            (*m_frames)[i] = FrameToImage(m_pending[i].data(), m_w, m_h, m_w * 4, m_k, m_rot, true);
            std::vector<uint8_t>().swap(m_pending[i]);   // each frame's raw copy gone as it goes
        });
        m_pending.clear();
    }

private:
    // Corners of `from`, followed into `to` (the same frame when kept now).
    void Seed(const Pyramid& from, const Pyramid& to) {
        m_ref = Corners(from);
        m_cur = m_ref;
        m_alive.assign(m_ref.size(), 1);
        if (&from != &to)
            for (size_t i = 0; i < m_ref.size(); ++i) {
                Pt b = m_ref[i];
                if (TrackChecked(from, to, m_ref[i], &b)) m_cur[i] = b;
                else m_alive[i] = 0;
            }
        m_seeded = int(m_ref.size());
    }

    void Offer(const uint8_t* px, int pitch, double t, double score, const Pyramid& p, double d) {
        if (m_haveCand && score <= m_candScore) return;
        m_cand.resize(size_t(m_w) * size_t(m_h) * 4);
        for (int y = 0; y < m_h; ++y)
            std::memcpy(m_cand.data() + size_t(y) * size_t(m_w) * 4, px + ptrdiff_t(y) * pitch,
                        size_t(m_w) * 4);
        m_candPyr = p;
        m_candScore = score;
        m_candTime = t;
        m_candD = d;
        m_haveCand = true;
    }

    void Emit(const Pyramid& now, int why) {
        // The pixels as they are; turned into an image -- downscaled and
        // rotated upright, 16 ms a frame -- in Finish, many at once. Done
        // here, one at a time, it was 25 s of decoding a 219 s room scan.
        m_pending.push_back(std::move(m_cand));
        m_cand.clear();
        m_info->times.push_back(m_candTime);
        m_info->sharpness.push_back(m_candScore);
        m_info->motion.push_back(m_candD);
        if (why == 1) ++m_info->byMotion;
        else if (why == 2) ++m_info->byTracking;
        else ++m_info->byTime;
        Seed(m_candPyr, now);
        m_refTime = m_candTime;
        m_haveCand = false;
        m_candScore = -1.0;
        m_first = false;
        // TOO MANY: every other frame goes and the step doubles, so spacing
        // still follows the motion, only coarser.
        if (m_o.maxFrames > 0 && int(m_pending.size()) > m_o.maxFrames) {
            auto thin = [](auto& v) {
                size_t o = 0;
                for (size_t i = 0; i < v.size(); i += 2) v[o++] = std::move(v[i]);
                v.resize(o);
            };
            thin(m_pending);
            thin(m_info->times);
            thin(m_info->sharpness);
            thin(m_info->motion);
            m_step *= 2.0;
            m_every *= 2.0;
            ++m_info->thinned;
        }
    }

    const VideoOptions& m_o;
    int m_w, m_h, m_rot, m_k = 1, m_kt = 1;
    double m_step = 0.1;
    double m_every = 0.0;   // the floor in time, seconds; 0 for none
    std::vector<double> m_recent;   // sharpness of the last ~second of frames
    static constexpr double kSharpFrac = 0.6;   // under this of it is blurred
    std::vector<Image>* m_frames;
    std::vector<std::vector<uint8_t>> m_pending;   // kept frames' pixels, BGRX, until Finish
    VideoInfo* m_info;
    bool m_started = false, m_first = true;
    Pyramid m_prev, m_candPyr;
    std::vector<Pt> m_ref, m_cur;
    std::vector<char> m_alive;
    int m_seeded = 0;
    double m_refTime = 0.0;
    std::vector<uint8_t> m_cand;
    bool m_haveCand = false;
    double m_candScore = -1.0, m_candTime = 0.0, m_candD = 0.0;
};

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

// NV12 -- the decoder's own output, w x h luma then (w/2) x (h/2)
// interleaved chroma, compact -- to top-down BGRX with X 0: BT.709, limited
// range, chroma repeated over its 2 x 2 pixels. That is what Media
// Foundation's own RGB32 conversion does for these clips, which declare no
// colour metadata: fitted on a frame of IMG_1528 against it, this agrees to
// within one level, on 3% of values and exactly on the rest (BT.601 was off
// by 20 levels, interpolated chroma by 87 at edges). Doing it here instead
// is the point: Media Foundation's conversion is single threaded and was
// 67 s of decoding a 219 s clip, 11 ms a frame against 2.3 for the decode.
void Nv12ToBgrx(const uint8_t* nv, int w, int h, uint8_t* out) {
    const uint8_t* uvPlane = nv + size_t(w) * size_t(h);
    // Serial: frames are converted many at once, a batch across the cores.
    for (int y = 0; y < h; ++y) {
        const uint8_t* yr = nv + size_t(y) * size_t(w);
        const uint8_t* uv = uvPlane + size_t(y / 2) * size_t(w);
        uint8_t* o = out + size_t(y) * size_t(w) * 4;
        for (int x = 0; x < w; ++x) {
            const double Y = (double(yr[x]) - 16.0) * (255.0 / 219.0);
            const double cb = (double(uv[(x & ~1)]) - 128.0) * (255.0 / 224.0);
            const double cr = (double(uv[(x & ~1) + 1]) - 128.0) * (255.0 / 224.0);
            const double r = Y + 1.5748 * cr;
            const double b = Y + 1.8556 * cb;
            const double g = (Y - 0.2126 * r - 0.0722 * b) / 0.7152;
            auto px8 = [](double v) {
                return uint8_t(v <= 0.0 ? 0 : v >= 255.0 ? 255 : int(v + 0.5));
            };
            o[4 * x + 0] = px8(b);
            o[4 * x + 1] = px8(g);
            o[4 * x + 2] = px8(r);
            o[4 * x + 3] = 0;
        }
    }
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

    // NV12 by motion -- converted here, in parallel; see Nv12ToBgrx -- and
    // RGB32 from the reader otherwise, or when NV12 is not on offer.
    MFCreateMediaType(&want);
    want->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    bool nv12 = opt.pick == VideoPick::Motion;
    hr = E_FAIL;
    if (nv12) {
        want->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12);
        hr = reader->SetCurrentMediaType(kStream, nullptr, want);
        nv12 = SUCCEEDED(hr);
    }
    if (!nv12) {
        want->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
        hr = reader->SetCurrentMediaType(kStream, nullptr, want);
    }
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
    info->motion.clear();
    info->byMotion = info->byTracking = info->byTime = info->thinned = 0;
    info->step = 0.0;
    info->byTimeSlots = opt.pick == VideoPick::Time;
    frames->clear();
    MotionPicker picker(opt, int(w), int(h), info->rotation, info->seconds, frames, info);
    const bool byMotion = opt.pick == VideoPick::Motion;

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
    auto flush = [&] {
        if (candSlot < 0) return;
        frames->push_back(FrameToImage(cand.data(), int(w), int(h), int(w) * 4, k,
                                       info->rotation, true));
        info->times.push_back(candTime);
        info->sharpness.push_back(candScore);
        candSlot = -1;
        candScore = -1.0;
    };

    // DECODING AND ANALYSIS OVERLAP, by motion, in batches. Measured on a
    // 219 s room scan, 6183 frames: decoding with Media Foundation's RGB
    // conversion 14.7 ms a frame, sharpness and motion tracking 11.6 ms, one
    // after the other -- 168 s. Now this thread decodes, NV12 and no more,
    // and hands over batches; the analyser converts, scores and builds the
    // tracking pyramids of a whole batch at once on every core, and only the
    // tracking itself goes frame by frame. The decisions are made on the
    // same pixels, to a level, so the frames kept are the same ones.
    struct Raw { std::vector<uint8_t> px; double t; };
    constexpr size_t kBatch = 32, kBatchesInFlight = 2;
    std::mutex qMtx;
    std::condition_variable qCv;
    std::deque<std::vector<Raw>> batches;
    std::vector<std::vector<uint8_t>> spare;
    bool decodeDone = false;
    std::thread analyser;
    if (byMotion) {
        analyser = std::thread([&] {
            // A batch prepared -- converted, scored, its pyramids built --
            // while the one before it is tracked: the two overlap, the
            // preparation on every core and the tracking on this thread.
            struct Prepared {
                std::vector<Raw> raw;
                std::vector<std::vector<uint8_t>> bgrx;
                std::vector<double> score;
                std::vector<Pyramid> pyr;
            };
            auto nextBatch = [&](std::vector<Raw>* out) {
                std::unique_lock<std::mutex> lk(qMtx);
                qCv.wait(lk, [&] { return !batches.empty() || decodeDone; });
                if (batches.empty()) return false;
                *out = std::move(batches.front());
                batches.pop_front();
                lk.unlock();
                qCv.notify_all();
                return true;
            };
            const int stride = int(w) * 4;
            auto prepare = [&](Prepared* batchIn) {
                Prepared& pb = *batchIn;
                const size_t n = pb.raw.size();
                pb.bgrx.resize(n);
                pb.score.assign(n, 0.0);
                pb.pyr.clear();
                pb.pyr.resize(n);
                const int fw = int(w), fh = int(h);
                ParallelFor(n, [&pb, &picker, fw, fh, stride, nv12](size_t i) {
                    const uint8_t* px = pb.raw[i].px.data();
                    if (nv12) {
                        pb.bgrx[i].resize(size_t(fw) * size_t(fh) * 4);
                        Nv12ToBgrx(px, fw, fh, pb.bgrx[i].data());
                        px = pb.bgrx[i].data();
                    }
                    pb.score[i] = FrameSharpness(px, fw, fh, stride);
                    pb.pyr[i] = picker.Prepare(px, stride);
                });
            };
            auto track = [&](Prepared* p) {
                for (size_t i = 0; i < p->raw.size(); ++i)
                    picker.Frame(nv12 ? p->bgrx[i].data() : p->raw[i].px.data(), stride,
                                 p->raw[i].t, p->score[i], std::move(p->pyr[i]));
                std::lock_guard<std::mutex> lk(qMtx);
                for (Raw& r : p->raw) spare.push_back(std::move(r.px));
                p->raw.clear();
            };

            Prepared cur, next;
            if (!nextBatch(&cur.raw)) return;
            prepare(&cur);
            for (;;) {
                const bool more = nextBatch(&next.raw);
                std::thread prep;
                if (more) prep = std::thread([&] { prepare(&next); });
                track(&cur);
                if (!more) return;
                prep.join();
                std::swap(cur, next);
            }
        });
    }
    std::vector<Raw> filling;
    auto handOver = [&] {
        if (filling.empty()) return;
        {
            std::unique_lock<std::mutex> lk(qMtx);
            qCv.wait(lk, [&] { return batches.size() < kBatchesInFlight; });
            batches.push_back(std::move(filling));
        }
        qCv.notify_all();
        filling.clear();
    };
    auto stopAnalyser = [&] {
        if (!analyser.joinable()) return;
        handOver();
        {
            std::lock_guard<std::mutex> lk(qMtx);
            decodeDone = true;
        }
        qCv.notify_all();
        analyser.join();
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
            stopAnalyser();
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
            // NV12's chroma plane follows the luma's rows, which a decoder
            // pads (1088 for 1080): found from the buffer's whole length.
            const uint8_t* uvPlane = nullptr;
            if (SUCCEEDED(buf->QueryInterface(IID_PPV_ARGS(&b2))) &&
                SUCCEEDED(b2->Lock2D(&scan0, &pitch))) {
                locked2d = true;
                DWORD clen = 0;
                if (nv12 && pitch > 0 && SUCCEEDED(b2->GetContiguousLength(&clen)))
                    uvPlane = scan0 + size_t(pitch) * (size_t(clen) * 2 / 3 / size_t(pitch));
            } else if (SUCCEEDED(buf->Lock(&raw, nullptr, &len))) {
                locked = true;
                scan0 = raw;
                if (nv12) {
                    pitch = LONG(w);
                    uvPlane = scan0 + size_t(w) * (size_t(len) * 2 / 3 / size_t(w));
                    if (size_t(len) < size_t(w) * size_t(h) * 3 / 2) scan0 = nullptr;
                } else {
                    pitch = BufferPitch(len, w, h);
                    if (len < DWORD(pitch) * h) scan0 = nullptr;   // truncated frame
                }
            }
            if (nv12 && !uvPlane) scan0 = nullptr;
            if (scan0 && byMotion) {
                Raw r;
                r.t = double(ts) * 1e-7;
                {
                    std::lock_guard<std::mutex> lk(qMtx);
                    if (!spare.empty()) {
                        r.px = std::move(spare.back());
                        spare.pop_back();
                    }
                }
                // Copied compact and top-down, whatever the stored layout.
                if (nv12) {
                    r.px.resize(size_t(w) * size_t(h) * 3 / 2);
                    for (UINT32 y = 0; y < h; ++y)
                        std::memcpy(r.px.data() + size_t(y) * w, scan0 + ptrdiff_t(y) * pitch, w);
                    uint8_t* uvOut = r.px.data() + size_t(w) * h;
                    for (UINT32 y = 0; y < h / 2; ++y)
                        std::memcpy(uvOut + size_t(y) * w, uvPlane + ptrdiff_t(y) * pitch, w);
                } else {
                    r.px.resize(size_t(w) * size_t(h) * 4);
                    for (UINT32 y = 0; y < h; ++y)
                        std::memcpy(r.px.data() + size_t(y) * w * 4, scan0 + ptrdiff_t(y) * pitch,
                                    size_t(w) * 4);
                }
                filling.push_back(std::move(r));
                if (filling.size() >= kBatch) handOver();
                ++info->decoded;
            } else if (scan0) {
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
    stopAnalyser();   // every queued frame analysed, in order
    if (byMotion) picker.Finish();
    else flush();
    cleanup();

    if (frames->empty()) {
        *err = "'" + path + "' decoded no frames";
        return false;
    }
    // A clip that barely moves -- a tripod shot, a slow drift -- has too
    // little parallax for a reconstruction either way, but equal time slots
    // at least give the rest of the chain something to try.
    if (byMotion && int(frames->size()) < opt.minFrames && info->decoded > opt.minFrames) {
        VideoOptions t = opt;
        t.pick = VideoPick::Time;
        t.frames = std::max(opt.minFrames, 30);
        return LoadVideoFrames(path, t, frames, info, err);
    }
    return true;
}

std::string VideoNote(const VideoInfo& vi, int kept) {
    char buf[240];
    int n = std::snprintf(buf, sizeof(buf), "%d frames kept of %d (%.1f s at %.0f fps, %dx%d%s)",
                          kept, vi.decoded, vi.seconds, vi.fps, vi.width, vi.height,
                          vi.rotation ? (", rotated " + std::to_string(vi.rotation) + "\xC2\xB0").c_str()
                                      : "");
    if (n > 0 && size_t(n) < sizeof(buf)) {
        if (vi.byTimeSlots)
            std::snprintf(buf + n, sizeof(buf) - size_t(n), "; by time");
        else
            std::snprintf(buf + n, sizeof(buf) - size_t(n),
                          "; by motion, a step of %.0f%%%s%s", 100.0 * vi.step,
                          vi.byTracking ? (", " + std::to_string(vi.byTracking) +
                                           " where the view was lost").c_str() : "",
                          vi.byTime ? (", " + std::to_string(vi.byTime) +
                                       " by the time floor").c_str() : "");
    }
    return buf;
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
