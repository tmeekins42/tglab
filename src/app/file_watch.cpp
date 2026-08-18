#include "file_watch.h"

#include <windows.h>

#include <fstream>
#include <sstream>

namespace tglab {

namespace {

bool StatFile(const std::string& path, uint64_t* writeTime, uint64_t* size) {
    WIN32_FILE_ATTRIBUTE_DATA fad{};
    if (!GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &fad)) return false;
    *writeTime = (uint64_t(fad.ftLastWriteTime.dwHighDateTime) << 32) | fad.ftLastWriteTime.dwLowDateTime;
    *size      = (uint64_t(fad.nFileSizeHigh) << 32) | fad.nFileSizeLow;
    return true;
}

double NowSeconds() {
    static LARGE_INTEGER freq = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f; }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return double(c.QuadPart) / double(freq.QuadPart);
}

} // namespace

void FileWatch::Watch(std::string path) {
    m_path = std::move(path);
    m_lastWrite = 0;
    m_lastSize  = 0;
    m_pending   = false;
    StatFile(m_path, &m_lastWrite, &m_lastSize);
}

bool FileWatch::Poll() {
    if (m_path.empty()) return false;

    uint64_t w = 0, s = 0;
    if (!StatFile(m_path, &w, &s)) return false;

    if (w != m_lastWrite || s != m_lastSize) {
        m_lastWrite = w;
        m_lastSize  = s;
        m_pending   = true;
        m_stableAt  = NowSeconds();
        return false;   // wait for the file to settle
    }

    // Debounce: fire once the size/time have been stable briefly, which avoids
    // reading a half-written file mid-save.
    if (m_pending && (NowSeconds() - m_stableAt) > 0.10) {
        m_pending = false;
        return true;
    }
    return false;
}

bool ReadTextFile(const std::string& path, std::string* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    *out = ss.str();
    return true;
}

} // namespace tglab
