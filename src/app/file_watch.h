// Polling file watcher.
//
// Deliberately poll-based rather than ReadDirectoryChangesW: editors write
// scripts via save-and-rename, which produces confusing event sequences, and
// a stat() once per frame is free at this scale.
#pragma once

#include <string>

namespace tglab {

class FileWatch {
public:
    void Watch(std::string path);

    // True when the file changed since the last call (debounced).
    bool Poll();

    const std::string& Path() const { return m_path; }

private:
    std::string m_path;
    uint64_t    m_lastWrite = 0;
    uint64_t    m_lastSize  = 0;
    double      m_stableAt  = 0.0;
    bool        m_pending   = false;
};

// Reads a whole file. Returns false if it could not be opened.
bool ReadTextFile(const std::string& path, std::string* out);

} // namespace tglab
