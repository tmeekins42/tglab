#include "lensfun.h"

#include <windows.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace tglab {
namespace {

// --- the database, as read ------------------------------------------------

struct DistEntry { float focal; int model; double t[3]; };       // 0 poly3, 1 poly5, 2 ptlens
struct TcaEntry  { float focal; int model; double t[6]; };       // 0 linear (kr kb), 1 poly3 (vr vb cr cb br bb)
struct VigEntry  { float focal, aperture, distance; double k[3]; };

struct LensEntry {
    std::string maker;
    std::vector<std::string> models;   // the primary name first, then translations
    std::vector<std::string> mounts;
    float crop   = 1.0f;
    float aspect = 1.5f;
    std::vector<DistEntry> dist;
    std::vector<TcaEntry>  tca;
    std::vector<VigEntry>  vig;
};

struct CameraEntry {
    std::string maker;
    std::vector<std::string> models;
    std::string mount;
    float crop = 0.0f;
};

struct Database {
    std::string dir;
    bool loaded = false;
    std::vector<LensEntry>   lenses;
    std::vector<CameraEntry> cameras;
};

std::mutex g_mutex;
Database   g_db;
std::string g_override;   // SetLensfunDirectory
bool        g_overridden = false;
// Answers already worked out, by everything that goes into one: a stage asks
// on every run, and matching walks every lens in the database.
std::map<std::string, LensProfile> g_cache;

// --- a minimal XML reader --------------------------------------------------
//
// Enough for lensfun's files: elements, attributes, text, comments and the
// five predefined entities. No DTDs, CDATA or namespaces -- the files use none.

std::string Decode(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '&') { o += s[i]; continue; }
        const size_t semi = s.find(';', i);
        if (semi == std::string::npos) { o += s[i]; continue; }
        const std::string e = s.substr(i + 1, semi - i - 1);
        if (e == "amp") o += '&';
        else if (e == "lt") o += '<';
        else if (e == "gt") o += '>';
        else if (e == "quot") o += '"';
        else if (e == "apos") o += '\'';
        else if (!e.empty() && e[0] == '#') {
            const long c = (e.size() > 1 && (e[1] == 'x' || e[1] == 'X'))
                               ? std::strtol(e.c_str() + 2, nullptr, 16)
                               : std::strtol(e.c_str() + 1, nullptr, 10);
            if (c > 0 && c < 128) o += char(c);   // names are ASCII; anything else is dropped
        } else { o += s[i]; continue; }
        i = semi;
    }
    return o;
}

std::string TrimWs(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

using Attrs = std::map<std::string, std::string>;

double AttrNum(const Attrs& a, const char* name, double def) {
    const auto it = a.find(name);
    return it == a.end() ? def : std::atof(it->second.c_str());
}

// "4:3" or "1.333".
float ParseAspect(const std::string& s) {
    const size_t colon = s.find(':');
    if (colon != std::string::npos) {
        const double a = std::atof(s.c_str()), b = std::atof(s.c_str() + colon + 1);
        return (a > 0 && b > 0) ? float(std::max(a, b) / std::min(a, b)) : 1.5f;
    }
    const double v = std::atof(s.c_str());
    return v > 0 ? float(v < 1 ? 1 / v : v) : 1.5f;
}

// One file's lenses and cameras, appended to `db`.
void ParseFile(const std::string& text, Database* db) {
    std::vector<std::string> stack;      // open element names
    LensEntry lens;
    CameraEntry cam;
    bool inLens = false, inCam = false;
    std::string textBuf;
    Attrs openAttrs;                     // the attributes of the innermost open element

    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        if (text[i] != '<') {
            const size_t lt = text.find('<', i);
            textBuf += text.substr(i, (lt == std::string::npos ? n : lt) - i);
            i = (lt == std::string::npos) ? n : lt;
            continue;
        }
        if (text.compare(i, 4, "<!--") == 0) {
            const size_t e = text.find("-->", i + 4);
            i = (e == std::string::npos) ? n : e + 3;
            continue;
        }
        if (text.compare(i, 2, "<?") == 0 || text.compare(i, 2, "<!") == 0) {
            const size_t e = text.find('>', i);
            i = (e == std::string::npos) ? n : e + 1;
            continue;
        }
        const size_t gt = text.find('>', i);
        if (gt == std::string::npos) break;
        std::string tag = text.substr(i + 1, gt - i - 1);
        i = gt + 1;

        if (!tag.empty() && tag[0] == '/') {          // a closing tag
            const std::string name = TrimWs(tag.substr(1));
            const std::string value = TrimWs(Decode(textBuf));
            textBuf.clear();
            const std::string parent = stack.size() >= 2 ? stack[stack.size() - 2] : std::string();
            if (name == "lens" && inLens) {
                if (!lens.models.empty()) db->lenses.push_back(std::move(lens));
                lens = LensEntry{};
                inLens = false;
            } else if (name == "camera" && inCam) {
                if (!cam.models.empty()) db->cameras.push_back(std::move(cam));
                cam = CameraEntry{};
                inCam = false;
            } else if (parent == "lens" && inLens) {
                if (name == "maker" && lens.maker.empty()) lens.maker = value;
                else if (name == "model") {
                    // The untranslated name is the one without a lang attribute.
                    if (openAttrs.count("lang")) lens.models.push_back(value);
                    else lens.models.insert(lens.models.begin(), value);
                } else if (name == "mount") lens.mounts.push_back(value);
                else if (name == "cropfactor") lens.crop = float(std::atof(value.c_str()));
                else if (name == "aspect-ratio") lens.aspect = ParseAspect(value);
            } else if (parent == "camera" && inCam) {
                if (name == "maker" && cam.maker.empty()) cam.maker = value;
                else if (name == "model") {
                    if (openAttrs.count("lang")) cam.models.push_back(value);
                    else cam.models.insert(cam.models.begin(), value);
                } else if (name == "mount" && cam.mount.empty()) cam.mount = value;
                else if (name == "cropfactor") cam.crop = float(std::atof(value.c_str()));
            }
            if (!stack.empty()) stack.pop_back();
            openAttrs.clear();
            continue;
        }

        const bool selfClose = !tag.empty() && tag.back() == '/';
        if (selfClose) tag.pop_back();
        size_t p = 0;
        while (p < tag.size() && !std::isspace(static_cast<unsigned char>(tag[p]))) ++p;
        const std::string name = tag.substr(0, p);
        Attrs attrs;
        while (p < tag.size()) {
            while (p < tag.size() && std::isspace(static_cast<unsigned char>(tag[p]))) ++p;
            const size_t eq = tag.find('=', p);
            if (eq == std::string::npos) break;
            const std::string key = TrimWs(tag.substr(p, eq - p));
            size_t q = eq + 1;
            while (q < tag.size() && std::isspace(static_cast<unsigned char>(tag[q]))) ++q;
            if (q >= tag.size() || (tag[q] != '"' && tag[q] != '\'')) break;
            const char quote = tag[q];
            const size_t end = tag.find(quote, q + 1);
            if (end == std::string::npos) break;
            attrs[key] = Decode(tag.substr(q + 1, end - q - 1));
            p = end + 1;
        }
        textBuf.clear();

        const std::string parent = stack.empty() ? std::string() : stack.back();
        if (name == "lens") { inLens = true; lens = LensEntry{}; }
        else if (name == "camera") { inCam = true; cam = CameraEntry{}; }
        else if (inLens && parent == "calibration") {
            const std::string model = attrs.count("model") ? attrs["model"] : std::string();
            const float focal = float(AttrNum(attrs, "focal", 0));
            if (name == "distortion" && focal > 0) {
                DistEntry d{focal, -1, {0, 0, 0}};
                if (model == "poly3")  { d.model = 0; d.t[0] = AttrNum(attrs, "k1", 0); }
                if (model == "poly5")  { d.model = 1; d.t[0] = AttrNum(attrs, "k1", 0); d.t[1] = AttrNum(attrs, "k2", 0); }
                if (model == "ptlens") { d.model = 2; d.t[0] = AttrNum(attrs, "a", 0); d.t[1] = AttrNum(attrs, "b", 0); d.t[2] = AttrNum(attrs, "c", 0); }
                if (d.model >= 0) lens.dist.push_back(d);
            } else if (name == "tca" && focal > 0) {
                TcaEntry t{focal, -1, {1, 1, 0, 0, 0, 0}};
                if (model == "linear") { t.model = 0; t.t[0] = AttrNum(attrs, "kr", 1); t.t[1] = AttrNum(attrs, "kb", 1); }
                if (model == "poly3") {
                    t.model = 1;
                    t.t[0] = AttrNum(attrs, "vr", 1); t.t[1] = AttrNum(attrs, "vb", 1);
                    t.t[2] = AttrNum(attrs, "cr", 0); t.t[3] = AttrNum(attrs, "cb", 0);
                    t.t[4] = AttrNum(attrs, "br", 0); t.t[5] = AttrNum(attrs, "bb", 0);
                }
                if (t.model >= 0) lens.tca.push_back(t);
            } else if (name == "vignetting" && focal > 0 && model == "pa") {
                lens.vig.push_back(VigEntry{focal, float(AttrNum(attrs, "aperture", 0)),
                                            float(AttrNum(attrs, "distance", 1000)),
                                            {AttrNum(attrs, "k1", 0), AttrNum(attrs, "k2", 0),
                                             AttrNum(attrs, "k3", 0)}});
            }
        }
        if (!selfClose) {
            stack.push_back(name);
            openAttrs = std::move(attrs);
        }
    }
}

bool DirHasXml(const std::string& dir) {
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((dir + "\\*.xml").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return false;
    FindClose(h);
    return true;
}

std::string SearchDirectory() {
    char env[MAX_PATH] = {};
    if (GetEnvironmentVariableA("TGLAB_LENSFUN_DIR", env, MAX_PATH) > 0 && DirHasXml(env))
        return env;
    char exe[MAX_PATH] = {};
    if (GetModuleFileNameA(nullptr, exe, MAX_PATH) > 0) {
        std::string dir(exe);
        for (int up = 0; up < 5; ++up) {
            const size_t slash = dir.find_last_of("/\\");
            if (slash == std::string::npos) break;
            dir = dir.substr(0, slash);
            // `lensfun` is where a release ships it, beside tglab.exe; the
            // repository's vendored copy is found from a build directory.
            for (const char* sub : {"\\lensfun", "\\third_party\\lensfun\\db"}) {
                const std::string p = dir + sub;
                if (DirHasXml(p)) return p;
            }
        }
    }
    return {};
}

// Loads on first use; the caller holds g_mutex.
const Database& Db() {
    if (g_db.loaded) return g_db;
    g_db = Database{};
    g_db.loaded = true;
    g_db.dir = g_overridden ? g_override : SearchDirectory();
    if (g_db.dir.empty()) return g_db;
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((g_db.dir + "\\*.xml").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return g_db;
    do {
        const std::string path = g_db.dir + "\\" + fd.cFileName;
        FILE* f = nullptr;
        if (fopen_s(&f, path.c_str(), "rb") != 0 || !f) continue;
        std::string text;
        char buf[1 << 16];
        size_t got;
        while ((got = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, got);
        std::fclose(f);
        ParseFile(text, &g_db);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    return g_db;
}

// --- matching names ---------------------------------------------------------
//
// EXIF and the database spell the same lens differently -- "RF24-105mm F4 L IS
// USM" against "Canon RF 24-105mm f/4L IS USM" -- so both are cut into
// lower-case tokens at every non-alphanumeric and at each letter/digit
// boundary ("2.8" stays one number), the makers' names dropped, and compared
// as sets. Numbers, and the roman numerals that mark a lens's version, must
// agree exactly: a 24-70 is never a 24-105, nor a GM II a GM.

std::vector<std::string> Tokens(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    int kind = 0;   // 1 letters, 2 digits
    auto flush = [&] { if (!cur.empty()) out.push_back(cur); cur.clear(); kind = 0; };
    for (size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (std::isalpha(c)) {
            if (kind != 1) flush();
            kind = 1;
            cur += char(std::tolower(c));
        } else if (std::isdigit(c) ||
                   (c == '.' && kind == 2 && i + 1 < s.size() && std::isdigit(static_cast<unsigned char>(s[i + 1])))) {
            if (kind != 2) flush();
            kind = 2;
            cur += char(c);
        } else {
            flush();
        }
    }
    flush();
    return out;
}

bool Strict(const std::string& t) {
    if (!t.empty() && std::isdigit(static_cast<unsigned char>(t[0]))) return true;
    return t == "ii" || t == "iii" || t == "iv" || t == "vi" || t == "vii";
}

// A number written "4.0" in one place is "4" in the other.
std::string Canon(const std::string& t) {
    if (t.empty() || !std::isdigit(static_cast<unsigned char>(t[0]))) return t;
    char b[32];
    std::snprintf(b, sizeof b, "%g", std::atof(t.c_str()));
    return b;
}

std::vector<std::string> NameTokens(const std::string& s, const std::vector<std::string>& drop) {
    std::vector<std::string> out;
    for (const std::string& t : Tokens(s)) {
        if (std::find(drop.begin(), drop.end(), t) != drop.end()) continue;
        out.push_back(Canon(t));
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// 0 when the strict tokens disagree, else the shared fraction of the larger set.
double Score(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    if (a.empty() || b.empty()) return 0;
    size_t common = 0;
    for (const std::string& t : a) {
        const bool inB = std::binary_search(b.begin(), b.end(), t);
        if (inB) ++common;
        else if (Strict(t)) return 0;
    }
    for (const std::string& t : b)
        if (Strict(t) && !std::binary_search(a.begin(), a.end(), t)) return 0;
    return double(common) / double(std::max(a.size(), b.size()));
}

std::vector<std::string> MakerTokens(const char* m1, const std::string& m2) {
    std::vector<std::string> d = Tokens(m1 ? m1 : "");
    for (const std::string& t : Tokens(m2)) d.push_back(t);
    // EXIF makers carry their corporate suffixes.
    for (const char* s : {"corporation", "corp", "company", "co", "ltd", "inc", "imaging", "optical"})
        d.push_back(s);
    return d;
}

// --- interpolating ----------------------------------------------------------

// The two entries either side of `x` in a sorted list of keys, and the weight
// of the upper one; outside the range, the end entry alone.
void Bracket(const std::vector<float>& keys, float x, int* lo, int* hi, double* w) {
    *lo = *hi = 0; *w = 0;
    if (keys.empty()) return;
    if (x <= keys.front()) { *lo = *hi = 0; return; }
    if (x >= keys.back()) { *lo = *hi = int(keys.size()) - 1; return; }
    for (size_t i = 0; i + 1 < keys.size(); ++i)
        if (x >= keys[i] && x <= keys[i + 1]) {
            *lo = int(i); *hi = int(i + 1);
            *w = keys[i + 1] > keys[i] ? double(x - keys[i]) / double(keys[i + 1] - keys[i]) : 0;
            return;
        }
}

// One distortion entry in lens_correct's units: f(R) as five coefficients.
// s = image half diagonal over the Hugin unit (half the calibration sensor's
// short side).
void DistToPoly(const DistEntry& e, double s, double out[5]) {
    for (int i = 0; i < 5; ++i) out[i] = 0;
    switch (e.model) {
    case 0: {   // poly3: r_d = r_u (1 - k1 + k1 r_u^2)
        const double d = 1 - e.t[0];
        out[0] = 1;
        out[2] = e.t[0] * s * s / (d * d * d);
        break;
    }
    case 1:     // poly5: r_d = r_u (1 + k1 r_u^2 + k2 r_u^4)
        out[0] = 1;
        out[2] = e.t[0] * s * s;
        out[4] = e.t[1] * s * s * s * s;
        break;
    case 2: {   // ptlens: r_d = r_u (a r^3 + b r^2 + c r + 1 - a - b - c)
        const double d = 1 - e.t[0] - e.t[1] - e.t[2];
        out[0] = 1;
        out[1] = e.t[2] * s / (d * d);
        out[2] = e.t[1] * s * s / (d * d * d);
        out[3] = e.t[0] * s * s * s / (d * d * d * d);
        break;
    }
    default: out[0] = 1;
    }
}

void TcaToPoly(const TcaEntry& e, double s, double out[2][3]) {
    if (e.model == 0) {
        out[0][0] = e.t[0]; out[0][1] = 0; out[0][2] = 0;
        out[1][0] = e.t[1]; out[1][1] = 0; out[1][2] = 0;
    } else {
        for (int c = 0; c < 2; ++c) {
            out[c][0] = e.t[c];
            out[c][1] = e.t[2 + c] * s;
            out[c][2] = e.t[4 + c] * s * s;
        }
    }
}

// The camera's crop factor from the database, by model name.
const CameraEntry* FindCamera(const Database& db, const char* make, const char* model) {
    if (!model || !*model) return nullptr;
    const CameraEntry* best = nullptr;
    double bestScore = 0;
    for (const CameraEntry& c : db.cameras) {
        const std::vector<std::string> drop = MakerTokens(make, c.maker);
        const std::vector<std::string> want = NameTokens(model, drop);
        for (const std::string& m : c.models) {
            const double sc = Score(want, NameTokens(m, drop));
            if (sc > bestScore) { bestScore = sc; best = &c; }
        }
    }
    // A camera's name is short and exact: anything less than all of it is a
    // different body.
    return bestScore >= 0.999 ? best : nullptr;
}

LensProfile Evaluate(const Database& db, const char* cameraMake, const char* cameraModel,
                     const char* lens, float focalMm, float aperture, float crop35);

} // namespace

LensProfile FindLensProfile(const char* cameraMake, const char* cameraModel,
                            const char* lens, float focalMm, float aperture,
                            float crop35) {
    LensProfile out;
    if (!lens || focalMm <= 0) return out;
    std::lock_guard<std::mutex> lock(g_mutex);
    const Database& db = Db();
    if (db.lenses.empty()) return out;
    char num[64];
    std::snprintf(num, sizeof num, "|%g|%g|%g", focalMm, aperture, crop35);
    const std::string key = std::string(cameraMake ? cameraMake : "") + "|" +
                            (cameraModel ? cameraModel : "") + "|" + lens + num;
    if (const auto it = g_cache.find(key); it != g_cache.end()) return it->second;
    out = Evaluate(db, cameraMake, cameraModel, lens, focalMm, aperture, crop35);
    g_cache[key] = out;
    return out;
}

namespace {

LensProfile Evaluate(const Database& db, const char* cameraMake, const char* cameraModel,
                     const char* lens, float focalMm, float aperture, float crop35) {
    LensProfile out;

    // The image's crop factor: the camera's entry, else the file's 35 mm
    // equivalent, else full frame.
    float crop = 0;
    const CameraEntry* cam = FindCamera(db, cameraMake, cameraModel);
    if (cam) {
        crop = cam->crop;
        out.camera = cam->models.front();
    }
    if (crop <= 0) crop = crop35 > 0 ? crop35 : 1.0f;

    // A compact names no lens -- it has only the one. lensfun files that lens
    // under the body's own mount ("sonyRX100III"), so a mount that exactly one
    // lens name carries is that lens.
    std::vector<const LensEntry*> cands;
    double bestScore = 0;
    if (!*lens) {
        if (!cam || cam->mount.empty()) return out;
        for (const LensEntry& L : db.lenses)
            if (std::find(L.mounts.begin(), L.mounts.end(), cam->mount) != L.mounts.end()) {
                if (!cands.empty() && cands.front()->models.front() != L.models.front()) return out;
                cands.push_back(&L);
            }
    }

    // Every entry under the best-matching name. The same lens is often
    // calibrated more than once, on bodies of different sizes.
    for (const LensEntry& L : db.lenses) {
        if (!*lens) break;
        const std::vector<std::string> drop = MakerTokens(cameraMake, L.maker);
        const std::vector<std::string> want = NameTokens(lens, drop);
        double sc = 0;
        for (const std::string& m : L.models) sc = std::max(sc, Score(want, NameTokens(m, drop)));
        if (sc < 0.6) continue;
        if (sc > bestScore + 1e-9) { bestScore = sc; cands.clear(); }
        if (sc >= bestScore - 1e-9) cands.push_back(&L);
    }
    if (cands.empty()) return out;

    // Of those, the calibration made on the smallest sensor that still covers
    // this one: a profile measured on full frame holds for an APS-C crop of
    // it, but one measured on APS-C says nothing about full frame's corners.
    const LensEntry* pick = nullptr;
    for (const LensEntry* L : cands)
        if (L->crop <= crop * 1.05f && (!pick || L->crop > pick->crop)) pick = L;
    if (!pick)
        for (const LensEntry* L : cands)
            if (!pick || L->crop < pick->crop) pick = L;

    out.valid = true;
    out.lens = pick->models.front();
    out.focal = focalMm;
    out.aperture = aperture;
    out.cropFactor = crop;
    out.calibCrop = pick->crop > 0 ? pick->crop : 1.0f;
    // R = 1 at this image's half diagonal; distortion and TCA have r = 1 at
    // half the calibration sensor's short side, vignetting at its half diagonal.
    const double sHugin = double(out.calibCrop) * std::hypot(double(pick->aspect), 1.0) / double(crop);
    const double sVig   = double(out.calibCrop) / double(crop);

    if (!pick->dist.empty()) {
        std::vector<DistEntry> e = pick->dist;
        std::sort(e.begin(), e.end(), [](const DistEntry& a, const DistEntry& b) { return a.focal < b.focal; });
        std::vector<float> keys;
        for (const DistEntry& d : e) keys.push_back(d.focal);
        int lo, hi; double w;
        Bracket(keys, focalMm, &lo, &hi, &w);
        double a[5], b[5];
        DistToPoly(e[size_t(lo)], sHugin, a);
        DistToPoly(e[size_t(hi)], sHugin, b);
        for (int i = 0; i < 5; ++i) out.dist[i] = a[i] + (b[i] - a[i]) * w;
        out.hasDistortion = true;
    }
    if (!pick->tca.empty()) {
        std::vector<TcaEntry> e = pick->tca;
        std::sort(e.begin(), e.end(), [](const TcaEntry& a, const TcaEntry& b) { return a.focal < b.focal; });
        std::vector<float> keys;
        for (const TcaEntry& t : e) keys.push_back(t.focal);
        int lo, hi; double w;
        Bracket(keys, focalMm, &lo, &hi, &w);
        double a[2][3], b[2][3];
        TcaToPoly(e[size_t(lo)], sHugin, a);
        TcaToPoly(e[size_t(hi)], sHugin, b);
        for (int c = 0; c < 2; ++c)
            for (int k = 0; k < 3; ++k) out.tca[c][k] = a[c][k] + (b[c][k] - a[c][k]) * w;
        out.hasTca = true;
    }
    if (!pick->vig.empty() && aperture > 0) {
        // Focused far: per (focal, aperture) the entry at the greatest
        // distance. Landscapes and architecture, where vignetting shows,
        // are mostly shot there, and EXIF rarely says otherwise.
        std::map<std::pair<float, float>, VigEntry> farthest;
        for (const VigEntry& v : pick->vig) {
            auto& slot = farthest[{v.focal, v.aperture}];
            if (slot.focal == 0 || v.distance > slot.distance) slot = v;
        }
        std::map<float, std::vector<VigEntry>> byFocal;
        for (const auto& kv : farthest) byFocal[kv.first.first].push_back(kv.second);
        // Within a focal, linear in 1/N between the bracketing apertures.
        auto atAperture = [&](const std::vector<VigEntry>& list, double k[3]) {
            std::vector<VigEntry> l = list;
            std::sort(l.begin(), l.end(), [](const VigEntry& a, const VigEntry& b) { return a.aperture < b.aperture; });
            std::vector<float> keys;
            for (const VigEntry& v : l) keys.push_back(v.aperture);
            int lo, hi; double w;
            Bracket(keys, aperture, &lo, &hi, &w);
            if (lo != hi) {
                const double ia = 1.0 / l[size_t(lo)].aperture, ib = 1.0 / l[size_t(hi)].aperture;
                w = (ia - 1.0 / aperture) / (ia - ib);
            }
            for (int i = 0; i < 3; ++i) k[i] = l[size_t(lo)].k[i] + (l[size_t(hi)].k[i] - l[size_t(lo)].k[i]) * w;
        };
        std::vector<float> keys;
        for (const auto& kv : byFocal) keys.push_back(kv.first);
        int lo, hi; double w;
        Bracket(keys, focalMm, &lo, &hi, &w);
        double a[3], b[3];
        atAperture(byFocal[keys[size_t(lo)]], a);
        atAperture(byFocal[keys[size_t(hi)]], b);
        const double s2 = sVig * sVig;
        out.vig[0] = (a[0] + (b[0] - a[0]) * w) * s2;
        out.vig[1] = (a[1] + (b[1] - a[1]) * w) * s2 * s2;
        out.vig[2] = (a[2] + (b[2] - a[2]) * w) * s2 * s2 * s2;
        out.hasVignetting = true;
    }
    return out;
}

} // namespace

std::string LensfunDirectory() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return Db().dir;
}

void SetLensfunDirectory(const std::string& dir) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_override = dir;
    g_overridden = !dir.empty();
    g_db = Database{};
    g_cache.clear();
}

int LensfunLensCount() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return int(Db().lenses.size());
}

int LensfunCameraCount() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return int(Db().cameras.size());
}

} // namespace tglab
