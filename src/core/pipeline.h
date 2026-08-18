// Pipeline: the recorded stage list plus the dirty-by-hash execution.
//
// Phase 1 (interpreter) RECORDS stages; it never runs an algorithm.
// Phase 2 (Execute) runs from the first stage whose inputs or parameters
// changed, reusing cached outputs before that point.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "algorithm.h"
#include "data.h"

namespace tglab {

// Reference to one output port of a stage. stage == -1 means a source image
// from the palette, with `port` as its palette index.
struct PortRef {
    int stage = -1;
    int port  = 0;

    bool operator==(const PortRef&) const = default;
};

struct Stage {
    std::unique_ptr<AlgorithmBase> algo;
    std::string          algoName;
    std::vector<PortRef> inputs;
    std::vector<Data>    outputs;
    uint64_t             paramHash = 0;
    bool                 valid     = false;   // outputs hold a usable result
    int                  line      = 0;       // for error messages
};

// A viewer declared by the script via display().
struct ViewerDecl {
    std::string name;
    PortRef     source;
};

class Pipeline {
public:
    // --- recording (phase 1) ---
    void Clear();
    int  AddStage(std::unique_ptr<AlgorithmBase> algo, std::string name,
                  std::vector<PortRef> inputs, size_t numOutputs, int line);
    void AddViewer(std::string name, PortRef src);

    // --- execution (phase 2) ---
    // `sources` are the palette images, indexed by PortRef::port when stage==-1.
    // Reuses cached outputs from `prev` where the stage is unchanged.
    bool Execute(std::vector<Data>* sources, Pipeline* prev, std::string* err);

    const Data* Resolve(PortRef r, const std::vector<Data>* sources) const;

    std::vector<Stage>&       Stages()       { return m_stages; }
    const std::vector<Stage>& Stages() const { return m_stages; }
    const std::vector<ViewerDecl>& Viewers() const { return m_viewers; }

private:
    // True if `a` and `b` are the same algorithm with the same wiring/params.
    static bool SameStage(const Stage& a, const Stage& b);

    std::vector<Stage>      m_stages;
    std::vector<ViewerDecl> m_viewers;
};

} // namespace tglab
