#include "nn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "../core/parallel.h"
#include "nn_gpu.h"

namespace tglab {
namespace nn {

// --- weight files ---------------------------------------------------------------

size_t Weight::Count() const {
    size_t n = 1;
    for (int d : shape) n *= size_t(d);
    return shape.empty() ? 0 : n;
}

namespace {

// IEEE half to float, including subnormals; the files carry no NaNs or
// infinities, but they decode correctly anyway.
float HalfToFloat(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {   // subnormal: normalise it
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            bits = sign | (exp << 23) | ((man & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

struct File {
    FILE* f = nullptr;
    ~File() { if (f) std::fclose(f); }
    bool Read(void* p, size_t n) { return std::fread(p, 1, n, f) == n; }
    bool Write(const void* p, size_t n) { return std::fwrite(p, 1, n, f) == n; }
};

} // namespace

bool WeightFile::Load(const std::string& path, std::string* err) {
    m_t.clear();
    File in;
    if (fopen_s(&in.f, path.c_str(), "rb") != 0 || !in.f) {
        *err = "cannot open " + path;
        return false;
    }
    char magic[4];
    uint32_t count = 0;
    if (!in.Read(magic, 4) || std::memcmp(magic, "TGW1", 4) != 0 || !in.Read(&count, 4)) {
        *err = path + " is not a .tgw weight file";
        return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
        uint16_t len = 0;
        uint8_t dtype = 0, rank = 0;
        std::string name;
        if (!in.Read(&len, 2)) break;
        name.resize(len);
        if (!in.Read(name.data(), len) || !in.Read(&dtype, 1) || !in.Read(&rank, 1)) break;
        Weight w;
        w.shape.resize(rank);
        bool ok = true;
        for (int& d : w.shape) ok = ok && in.Read(&d, 4) && d >= 0;
        if (!ok || rank == 0 || dtype > 1) { *err = path + ": bad entry '" + name + "'"; return false; }
        w.data.resize(w.Count());
        if (dtype == 0) {
            ok = in.Read(w.data.data(), w.data.size() * 4);
        } else {
            std::vector<uint16_t> h(w.data.size());
            ok = in.Read(h.data(), h.size() * 2);
            for (size_t k = 0; k < h.size(); ++k) w.data[k] = HalfToFloat(h[k]);
        }
        if (!ok) { *err = path + ": truncated at '" + name + "'"; return false; }
        m_t[name] = std::move(w);
    }
    if (m_t.size() != count) { *err = path + ": truncated"; return false; }
    return true;
}

bool WeightFile::Save(const std::string& path, std::string* err) const {
    File out;
    if (fopen_s(&out.f, path.c_str(), "wb") != 0 || !out.f) {
        *err = "cannot write " + path;
        return false;
    }
    const uint32_t count = uint32_t(m_t.size());
    bool ok = out.Write("TGW1", 4) && out.Write(&count, 4);
    for (const auto& [name, w] : m_t) {
        const uint16_t len = uint16_t(name.size());
        const uint8_t dtype = 0, rank = uint8_t(w.shape.size());
        ok = ok && out.Write(&len, 2) && out.Write(name.data(), len) &&
             out.Write(&dtype, 1) && out.Write(&rank, 1) &&
             out.Write(w.shape.data(), w.shape.size() * 4) &&
             out.Write(w.data.data(), w.data.size() * 4);
    }
    if (!ok) *err = "write failed: " + path;
    return ok;
}

const Weight* WeightFile::Find(const std::string& name) const {
    const auto it = m_t.find(name);
    return it == m_t.end() ? nullptr : &it->second;
}

std::string FindModelFile(const std::string& file) {
    auto exists = [](const std::string& p) {
        return GetFileAttributesA(p.c_str()) != INVALID_FILE_ATTRIBUTES;
    };
    char env[MAX_PATH] = {};
    if (GetEnvironmentVariableA("TGLAB_MODELS", env, MAX_PATH) > 0) {
        const std::string p = std::string(env) + "\\" + file;
        if (exists(p)) return p;
    }
    char exe[MAX_PATH] = {};
    if (GetModuleFileNameA(nullptr, exe, MAX_PATH) > 0) {
        std::string dir(exe);
        for (int up = 0; up < 4; ++up) {
            const size_t slash = dir.find_last_of("/\\");
            if (slash == std::string::npos) break;
            dir = dir.substr(0, slash);
            const std::string p = dir + "\\models\\" + file;
            if (exists(p)) return p;
        }
    }
    const std::string p = "models\\" + file;
    return exists(p) ? p : std::string();
}

bool Conv::From(const WeightFile& f, const std::string& name, std::string* err) {
    const Weight* wt = f.Find(name + ".weight");
    const Weight* bs = f.Find(name + ".bias");
    if (!wt || wt->shape.size() != 4 || wt->shape[2] != wt->shape[3]) {
        *err = "missing or malformed " + name + ".weight";
        return false;
    }
    cout = wt->shape[0];
    k    = wt->shape[2];
    const int cinPerGroup = wt->shape[1];
    // Depthwise when each output channel sees one input channel; the networks
    // here use only that and dense convolutions, so nothing in between.
    groups = (cinPerGroup == 1 && cout > 1) ? cout : 1;
    cin    = cinPerGroup * groups;
    w = wt->data;
    if (bs) {
        if (bs->Count() != size_t(cout)) { *err = name + ".bias has the wrong size"; return false; }
        b = bs->data;
    } else {
        b.assign(size_t(cout), 0.0f);
    }
    gw.Release();
    gb.Release();
    return true;
}

bool Conv::FromLinear(const WeightFile& f, const std::string& name, std::string* err) {
    const Weight* wt = f.Find(name + ".weight");
    const Weight* bs = f.Find(name + ".bias");
    if (!wt || wt->shape.size() != 2) {
        *err = "missing or malformed " + name + ".weight";
        return false;
    }
    cout = wt->shape[0];
    cin = wt->shape[1];
    k = 1;
    groups = 1;
    w = wt->data;   // [out][in] is [cout][cin][1][1]
    if (bs && bs->Count() != size_t(cout)) { *err = name + ".bias has the wrong size"; return false; }
    if (bs) b = bs->data;
    else    b.assign(size_t(cout), 0.0f);
    gw.Release();
    gb.Release();
    return true;
}

bool Vec::From(const WeightFile& f, const std::string& name, std::string* err) {
    const Weight* wt = f.Find(name);
    if (!wt) { *err = "missing " + name; return false; }
    v = wt->data;
    g.Release();
    return true;
}

// --- the CPU reference ----------------------------------------------------------

namespace {

void CpuConv(const Tensor& in, const Conv& cv, bool relu, Tensor* out, int outOff) {
    const int H = in.h, W = in.w, k = cv.k, r = k / 2;
    const int cinG = cv.cin / cv.groups, coutG = cv.cout / cv.groups;
    const size_t P = in.Plane();
    ParallelFor(size_t(cv.cout), [&](size_t oc) {
        float* o = &out->cpu[(size_t(outOff) + oc) * P];
        std::fill(o, o + P, cv.b[oc]);
        const int g = int(oc) / coutG;
        for (int ic = 0; ic < cinG; ++ic) {
            const float* src = &in.cpu[size_t(g * cinG + ic) * P];
            const float* wk  = &cv.w[(oc * size_t(cinG) + size_t(ic)) * size_t(k * k)];
            for (int ky = 0; ky < k; ++ky)
                for (int kx = 0; kx < k; ++kx) {
                    const float wv = wk[ky * k + kx];
                    const int dy = ky - r, dx = kx - r;
                    const int x0 = std::max(0, -dx), x1 = std::min(W, W - dx);
                    for (int y = std::max(0, -dy); y < std::min(H, H - dy); ++y) {
                        float* orow = o + size_t(y) * size_t(W);
                        const float* srow = src + size_t(y + dy) * size_t(W) + dx;
                        for (int x = x0; x < x1; ++x) orow[x] += wv * srow[x];
                    }
                }
        }
        if (relu)
            for (size_t i = 0; i < P; ++i) o[i] = std::max(o[i], 0.0f);
    });
}

// PyTorch's cubic convolution with A = -0.75 (upsample_bicubic2d).
inline float Cubic1(float x, float A) { return ((A + 2) * x - (A + 3)) * x * x + 1; }
inline float Cubic2(float x, float A) { return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A; }
inline void CubicCoeffs(float t, float c[4]) {
    const float A = -0.75f;
    c[0] = Cubic2(t + 1.0f, A);
    c[1] = Cubic1(t, A);
    c[2] = Cubic1(1.0f - t, A);
    c[3] = Cubic2(2.0f - t, A);
}

void CpuResize(const Tensor& in, int inOff, int count, Interp mode, Tensor* out, int outOff) {
    const int ih = in.h, iw = in.w, oh = out->h, ow = out->w;
    const float sy = float(ih) / float(oh), sx = float(iw) / float(ow);
    ParallelFor(size_t(count), [&](size_t c) {
        const float* src = &in.cpu[(size_t(inOff) + c) * in.Plane()];
        float* dst = &out->cpu[(size_t(outOff) + c) * out->Plane()];
        auto at = [&](int y, int x) {
            return src[size_t(std::clamp(y, 0, ih - 1)) * size_t(iw) +
                       size_t(std::clamp(x, 0, iw - 1))];
        };
        for (int y = 0; y < oh; ++y)
            for (int x = 0; x < ow; ++x) {
                float v;
                if (mode == Interp::Bilinear) {
                    // Negative source positions clamp to 0: PyTorch's
                    // area_pixel_compute_source_index for the linear modes.
                    const float fy = std::max(0.0f, sy * (float(y) + 0.5f) - 0.5f);
                    const float fx = std::max(0.0f, sx * (float(x) + 0.5f) - 0.5f);
                    const int y0 = int(fy), x0 = int(fx);
                    const int y1 = y0 + (y0 < ih - 1 ? 1 : 0), x1 = x0 + (x0 < iw - 1 ? 1 : 0);
                    const float ly = fy - float(y0), lx = fx - float(x0);
                    v = (1 - ly) * ((1 - lx) * at(y0, x0) + lx * at(y0, x1)) +
                        ly * ((1 - lx) * at(y1, x0) + lx * at(y1, x1));
                } else {
                    // Cubic is not clamped, and its taps clamp to the edge.
                    const float fy = sy * (float(y) + 0.5f) - 0.5f;
                    const float fx = sx * (float(x) + 0.5f) - 0.5f;
                    const int y0 = int(std::floor(fy)), x0 = int(std::floor(fx));
                    float cy[4], cx[4];
                    CubicCoeffs(fy - float(y0), cy);
                    CubicCoeffs(fx - float(x0), cx);
                    v = 0.0f;
                    for (int j = 0; j < 4; ++j) {
                        float row = 0.0f;
                        for (int i = 0; i < 4; ++i) row += cx[i] * at(y0 - 1 + j, x0 - 1 + i);
                        v += cy[j] * row;
                    }
                }
                dst[size_t(y) * size_t(ow) + size_t(x)] = v;
            }
    });
}

} // namespace

// --- the engine -----------------------------------------------------------------

Engine::Engine(ComputeContext* gpu) : m_gpu(gpu) {
    if (m_gpu) m_k = std::make_unique<GpuKernels>(m_gpu);
}

Engine::~Engine() = default;

bool Engine::Alloc(Tensor* t, int c, int h, int w) {
    if (c <= 0 || h <= 0 || w <= 0) return Fail("empty tensor shape");
    t->c = c; t->h = h; t->w = w;
    if (!m_gpu) {
        t->cpu.resize(t->Count());
        return true;
    }
    t->cpu.clear();
    const uint64_t bytes = uint64_t(t->Count()) * 4u;
    // Kept when it fits without wasting more than half of itself, as the
    // pool would choose; otherwise swapped for a pooled one.
    if (t->gpu.Valid() && t->gpu.bytes >= bytes && t->gpu.bytes <= 2 * bytes) return true;
    t->Recycle();
    t->pool = m_gpu;
    if (!m_gpu->AcquireBuffer(bytes, &t->gpu)) return Fail("could not allocate a GPU tensor");
    return true;
}

bool Engine::Upload(const std::vector<float>& v, int c, int h, int w, Tensor* t) {
    if (v.size() != size_t(c) * size_t(h) * size_t(w)) return Fail("upload size mismatch");
    if (!Alloc(t, c, h, w)) return false;
    if (!m_gpu) {
        t->cpu = v;
        return true;
    }
    if (!m_gpu->UploadBuffer(v.data(), v.size() * 4, &t->gpu)) return Fail("tensor upload failed");
    return true;
}

bool Engine::Download(const Tensor& t, std::vector<float>* v) {
    if (!m_gpu) {
        *v = t.cpu;
        return true;
    }
    v->resize(t.Count());
    if (!m_gpu->ReadbackBuffer(t.gpu, v->data(), v->size() * 4))
        return Fail("tensor readback failed");
    return true;
}

bool Engine::Conv2d(const Tensor& in, const Conv& cv, bool relu, Tensor* out, int outOff) {
    if (in.c != cv.cin) return Fail("conv input has " + std::to_string(in.c) +
                                    " channels, expected " + std::to_string(cv.cin));
    if (out->h != in.h || out->w != in.w || outOff + cv.cout > out->c)
        return Fail("conv output tensor has the wrong shape");
    if (m_gpu) {
        const std::string op = (cv.groups > 1 ? "dw" : "conv") + std::to_string(cv.k) + "x" +
                               std::to_string(cv.k);
        return Timed(op.c_str(), [&] { return m_k->Conv2d(in, cv, relu, out, outOff, &m_err); });
    }
    CpuConv(in, cv, relu, out, outOff);
    return true;
}

bool Engine::MaxPool2(const Tensor& in, Tensor* out) {
    if (!Alloc(out, in.c, in.h / 2, in.w / 2)) return false;
    if (m_gpu) return Timed("pool", [&] { return m_k->MaxPool2(in, out, &m_err); });
    ParallelFor(size_t(in.c), [&](size_t c) {
        const float* s = &in.cpu[c * in.Plane()];
        float* d = &out->cpu[c * out->Plane()];
        for (int y = 0; y < out->h; ++y)
            for (int x = 0; x < out->w; ++x) {
                const float* p = s + size_t(2 * y) * size_t(in.w) + size_t(2 * x);
                d[size_t(y) * size_t(out->w) + size_t(x)] =
                    std::max(std::max(p[0], p[1]), std::max(p[in.w], p[in.w + 1]));
            }
    });
    return true;
}

bool Engine::Resize(const Tensor& in, int inOff, int count, Interp mode, Tensor* out,
                    int outOff) {
    if (inOff + count > in.c || outOff + count > out->c) return Fail("resize channel range");
    if (m_gpu)
        return Timed(mode == Interp::Bicubic ? "bicubic" : "bilinear",
                     [&] { return m_k->Resize(in, inOff, count, mode, out, outOff, &m_err); });
    CpuResize(in, inOff, count, mode, out, outOff);
    return true;
}

bool Engine::Copy(const Tensor& src, int srcOff, int count, Tensor* dst, int dstOff) {
    if (src.h != dst->h || src.w != dst->w || srcOff + count > src.c || dstOff + count > dst->c)
        return Fail("copy shape mismatch");
    if (m_gpu)
        return Timed("copy", [&] { return m_k->CopyAcc(src, srcOff, count, dst, dstOff, false, &m_err); });
    std::copy_n(&src.cpu[size_t(srcOff) * src.Plane()], size_t(count) * src.Plane(),
                &dst->cpu[size_t(dstOff) * dst->Plane()]);
    return true;
}

bool Engine::Accumulate(const Tensor& src, int srcOff, int count, Tensor* dst, int dstOff) {
    if (src.h != dst->h || src.w != dst->w || srcOff + count > src.c || dstOff + count > dst->c)
        return Fail("accumulate shape mismatch");
    if (m_gpu)
        return Timed("accumulate", [&] { return m_k->CopyAcc(src, srcOff, count, dst, dstOff, true, &m_err); });
    const float* s = &src.cpu[size_t(srcOff) * src.Plane()];
    float* d = &dst->cpu[size_t(dstOff) * dst->Plane()];
    for (size_t i = 0; i < size_t(count) * src.Plane(); ++i) d[i] += s[i];
    return true;
}

bool Engine::AddScaled(const Tensor& a, const Tensor& b, float s, Tensor* out) {
    if (a.c != b.c || a.h != b.h || a.w != b.w) return Fail("add shape mismatch");
    // A buffer cannot be read as an SRV and written as a UAV in one dispatch.
    if (out == &a || out == &b) return Fail("AddScaled cannot write in place");
    if (!Alloc(out, a.c, a.h, a.w)) return false;
    if (m_gpu) return Timed("add", [&] { return m_k->AddScaled(a, b, s, out, &m_err); });
    for (size_t i = 0; i < a.Count(); ++i) out->cpu[i] = (a.cpu[i] + b.cpu[i]) * s;
    return true;
}

bool Engine::SampleBilinear(const Tensor& map, const std::vector<float>& xy, Tensor* out) {
    const int n = int(xy.size() / 2);
    if (n <= 0) return Fail("no points to sample");
    if (!m_gpu) {
        if (!Alloc(out, 1, n, map.c)) return false;
        ParallelFor(size_t(n), [&](size_t i) {
            float* o = &out->cpu[i * size_t(map.c)];
            // grid_sample's unnormalisation for align_corners=False.
            const float ix = ((xy[2 * i] + 1.0f) * float(map.w) - 1.0f) * 0.5f;
            const float iy = ((xy[2 * i + 1] + 1.0f) * float(map.h) - 1.0f) * 0.5f;
            const int x0 = int(std::floor(ix)), y0 = int(std::floor(iy));
            const float fx = ix - float(x0), fy = iy - float(y0);
            const float wt[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
            const int xs[4] = {x0, x0 + 1, x0, x0 + 1}, ys[4] = {y0, y0, y0 + 1, y0 + 1};
            for (int c = 0; c < map.c; ++c) {
                float v = 0.0f;
                for (int k = 0; k < 4; ++k)
                    if (xs[k] >= 0 && xs[k] < map.w && ys[k] >= 0 && ys[k] < map.h)
                        v += wt[k] * map.cpu[(size_t(c) * size_t(map.h) + size_t(ys[k])) *
                                                 size_t(map.w) + size_t(xs[k])];
                o[c] = v;
            }
        });
        return true;
    }
    Tensor pts;
    if (!Upload(xy, 1, n, 2, &pts)) return false;
    if (!Alloc(out, 1, n, map.c)) return false;
    return Timed("sample", [&] { return m_k->SampleBilinear(map, pts, n, out, &m_err); });
}

// --- for transformers ------------------------------------------------------------

bool Engine::LayerNormGelu(const Tensor& in, const Vec& gamma, const Vec& beta, Tensor* out) {
    if (gamma.v.size() != size_t(in.c) || beta.v.size() != size_t(in.c))
        return Fail("layer norm size mismatch");
    if (out == &in) return Fail("LayerNormGelu cannot write in place");
    if (!Alloc(out, in.c, in.h, in.w)) return false;
    if (m_gpu)
        return Timed("layernorm", [&] { return m_k->LayerNormGelu(in, gamma, beta, out, &m_err); });
    const size_t P = in.Plane(), C = size_t(in.c);
    ParallelFor(P, [&](size_t p) {
        double mean = 0.0, var = 0.0;
        for (size_t c = 0; c < C; ++c) mean += in.cpu[c * P + p];
        mean /= double(C);
        for (size_t c = 0; c < C; ++c) {
            const double d = in.cpu[c * P + p] - mean;
            var += d * d;
        }
        const double inv = 1.0 / std::sqrt(var / double(C) + 1e-5);
        for (size_t c = 0; c < C; ++c) {
            const float x = float((in.cpu[c * P + p] - mean) * inv) * gamma.v[c] + beta.v[c];
            out->cpu[c * P + p] = 0.5f * x * (1.0f + std::erf(x * 0.70710678f));
        }
    });
    return true;
}

bool Engine::Rotary(Tensor* t, int off, int count, int headDim, const Tensor& pos,
                    const Vec& freq) {
    if (off + count > t->c || count % headDim || headDim % 2 || pos.c != 2 ||
        pos.Plane() != t->Plane() || freq.v.size() != size_t(headDim))
        return Fail("rotary encoding shape mismatch");
    if (m_gpu)
        return Timed("rotary", [&] { return m_k->Rotary(t, off, count, headDim, pos, freq, &m_err); });
    const size_t P = t->Plane();
    ParallelFor(P, [&](size_t p) {
        const float x = pos.cpu[p], y = pos.cpu[P + p];
        for (int j = 0; j < headDim / 2; ++j) {
            const float a = freq.v[size_t(2 * j)] * x + freq.v[size_t(2 * j + 1)] * y;
            const float cs = std::cos(a), sn = std::sin(a);
            for (int c = off + 2 * j; c < off + count; c += headDim) {
                float& u = t->cpu[size_t(c) * P + p];
                float& v = t->cpu[size_t(c + 1) * P + p];
                const float u0 = u, v0 = v;
                u = u0 * cs - v0 * sn;
                v = v0 * cs + u0 * sn;
            }
        }
    });
    return true;
}

bool Engine::Attention(const Tensor& q, int qOff, const Tensor& k, int kOff, const Tensor& v,
                       int vOff, int heads, int headDim, Tensor* out, int outOff) {
    const int width = heads * headDim;
    if (qOff + width > q.c || kOff + width > k.c || vOff + width > v.c ||
        outOff + width > out->c || k.Plane() != v.Plane() || out->Plane() != q.Plane())
        return Fail("attention shape mismatch");
    if (out == &q || out == &k || out == &v) return Fail("Attention cannot write in place");
    if (m_gpu)
        return Timed("attention", [&] {
            return m_k->Attention(q, qOff, k, kOff, v, vOff, heads, headDim, out, outOff, &m_err);
        });
    const size_t nq = q.Plane(), nk = k.Plane();
    const float scale = 1.0f / std::sqrt(float(headDim));
    ParallelFor(size_t(heads) * nq, [&](size_t hi) {
        const int h = int(hi / nq);
        const size_t i = hi % nq;
        std::vector<float> s(nk);
        float mx = -1e30f;
        for (size_t j = 0; j < nk; ++j) {
            float d = 0.0f;
            for (int c = 0; c < headDim; ++c)
                d += q.cpu[size_t(qOff + h * headDim + c) * nq + i] *
                     k.cpu[size_t(kOff + h * headDim + c) * nk + j];
            s[j] = d * scale;
            mx = std::max(mx, s[j]);
        }
        double sum = 0.0;
        for (float& e : s) sum += (e = std::exp(e - mx));
        for (int c = 0; c < headDim; ++c) {
            double acc = 0.0;
            const float* vr = &v.cpu[size_t(vOff + h * headDim + c) * nk];
            for (size_t j = 0; j < nk; ++j) acc += double(s[j]) * vr[j];
            out->cpu[size_t(outOff + h * headDim + c) * nq + i] = float(acc / sum);
        }
    });
    return true;
}

bool Engine::MatMulTN(const Tensor& a, const Tensor& b, float scale, Tensor* out) {
    if (a.c != b.c) return Fail("matmul depth mismatch");
    const int m = int(a.Plane()), n = int(b.Plane());
    if (!Alloc(out, 1, m, n)) return false;
    if (m_gpu) return Timed("matmul", [&] { return m_k->MatMulTN(a, b, scale, out, &m_err); });
    ParallelFor(size_t(m), [&](size_t i) {
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            for (int d = 0; d < a.c; ++d)
                s += a.cpu[size_t(d) * size_t(m) + i] * b.cpu[size_t(d) * size_t(n) + size_t(j)];
            out->cpu[i * size_t(n) + size_t(j)] = s * scale;
        }
    });
    return true;
}

bool Engine::DualSoftmaxBest(const Tensor& sim, std::vector<int>* best0, std::vector<float>* p0,
                             std::vector<int>* best1) {
    if (sim.c != 1) return Fail("similarity must be 1 x m x n");
    if (m_gpu)
        return Timed("dualsoftmax", [&] { return m_k->DualSoftmaxBest(sim, best0, p0, best1, &m_err); });
    const int m = sim.h, n = sim.w;
    auto at = [&](int i, int j) { return sim.cpu[size_t(i) * size_t(n) + size_t(j)]; };
    // The log normalisers of the row and the column softmaxes; then
    // P[i][j] = exp(2 s - rowLse[i] - colLse[j]), so the best of a row is the
    // largest 2 s - colLse and the best of a column the largest 2 s - rowLse.
    std::vector<float> rowLse, colLse;
    rowLse.resize(size_t(m));
    colLse.resize(size_t(n));
    ParallelFor(size_t(m), [&](size_t i) {
        float mx = -1e30f;
        for (int j = 0; j < n; ++j) mx = std::max(mx, at(int(i), j));
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += std::exp(double(at(int(i), j) - mx));
        rowLse[i] = mx + float(std::log(s));
    });
    ParallelFor(size_t(n), [&](size_t j) {
        float mx = -1e30f;
        for (int i = 0; i < m; ++i) mx = std::max(mx, at(i, int(j)));
        double s = 0.0;
        for (int i = 0; i < m; ++i) s += std::exp(double(at(i, int(j)) - mx));
        colLse[j] = mx + float(std::log(s));
    });
    best0->assign(size_t(m), -1);
    p0->assign(size_t(m), 0.0f);
    best1->assign(size_t(n), -1);
    ParallelFor(size_t(m), [&](size_t i) {
        float bv = -1e30f;
        for (int j = 0; j < n; ++j) {
            const float e = 2.0f * at(int(i), j) - colLse[size_t(j)];
            if (e > bv) { bv = e; (*best0)[i] = j; }
        }
        (*p0)[i] = std::exp(bv - rowLse[i]);
    });
    ParallelFor(size_t(n), [&](size_t j) {
        float bv = -1e30f;
        for (int i = 0; i < m; ++i) {
            const float e = 2.0f * at(i, int(j)) - rowLse[size_t(i)];
            if (e > bv) { bv = e; (*best1)[j] = i; }
        }
    });
    return true;
}

bool Engine::Flush() {
    if (!m_gpu) return true;
    return m_gpu->Flush(&m_err);
}

} // namespace nn
} // namespace tglab
