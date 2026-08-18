#include "pipeline.h"

#include <cassert>

namespace tglab {

void Pipeline::Clear() {
    m_stages.clear();
    m_viewers.clear();
}

int Pipeline::AddStage(std::unique_ptr<AlgorithmBase> algo, std::string name,
                       std::vector<PortRef> inputs, size_t numOutputs, int line) {
    Stage s;
    s.paramHash = algo->ParamHash();
    s.algo      = std::move(algo);
    s.algoName  = std::move(name);
    s.inputs    = std::move(inputs);
    s.outputs.resize(numOutputs);
    s.line      = line;
    m_stages.push_back(std::move(s));
    return int(m_stages.size()) - 1;
}

void Pipeline::AddViewer(std::string name, PortRef src) {
    // Re-declaring the same viewer name updates it rather than duplicating.
    for (ViewerDecl& v : m_viewers) {
        if (v.name == name) {
            v.source = src;
            return;
        }
    }
    m_viewers.push_back({std::move(name), src});
}

const Data* Pipeline::Resolve(PortRef r, const std::vector<Data>* sources) const {
    if (r.stage < 0) {
        if (!sources || r.port < 0 || size_t(r.port) >= sources->size()) return nullptr;
        return &(*sources)[size_t(r.port)];
    }
    if (size_t(r.stage) >= m_stages.size()) return nullptr;
    const Stage& s = m_stages[size_t(r.stage)];
    if (r.port < 0 || size_t(r.port) >= s.outputs.size()) return nullptr;
    return &s.outputs[size_t(r.port)];
}

bool Pipeline::SameStage(const Stage& a, const Stage& b) {
    return a.algoName == b.algoName && a.inputs == b.inputs &&
           a.paramHash == b.paramHash && a.outputs.size() == b.outputs.size();
}

bool Pipeline::Execute(std::vector<Data>* sources, Pipeline* prev, std::string* err) {
    // Find the first stage that differs from the previous run. Everything
    // before it can reuse its cached output. Comparing by hash means there is
    // no dirty flag to forget to set.
    size_t firstDirty = 0;
    if (prev) {
        const size_t n = std::min(m_stages.size(), prev->m_stages.size());
        while (firstDirty < n &&
               prev->m_stages[firstDirty].valid &&
               SameStage(m_stages[firstDirty], prev->m_stages[firstDirty])) {
            // Steal the cached outputs — Data is move-only, and the previous
            // pipeline is discarded right after this call.
            m_stages[firstDirty].outputs = std::move(prev->m_stages[firstDirty].outputs);
            m_stages[firstDirty].valid   = true;
            ++firstDirty;
        }
    }

    for (size_t i = firstDirty; i < m_stages.size(); ++i) {
        Stage& s = m_stages[i];
        s.valid = false;

        // Gather inputs.
        std::vector<const Data*> in;
        in.reserve(s.inputs.size());
        for (const PortRef& r : s.inputs) {
            const Data* d = Resolve(r, sources);
            if (!d || TypeOf(*d) == DataType::None) {
                *err = "line " + std::to_string(s.line) + ": '" + s.algoName +
                       "' has an input that produced no data";
                return false;
            }
            in.push_back(d);
        }

        // Allocate outputs. M1: every port is an image sized from input 0,
        // with the format resolved from the port's FormatSpec.
        const PortList outPorts = s.algo->Outputs();
        ImageDesc base{};
        if (!in.empty() && std::holds_alternative<Image>(*in[0]))
            base = std::get<Image>(*in[0]).Desc();

        for (size_t p = 0; p < s.outputs.size(); ++p) {
            ImageDesc d = base;
            switch (outPorts[p].format) {
                case FormatSpec::RGBA8:       d.format = Format::RGBA8;   break;
                case FormatSpec::R32F:        d.format = Format::R32F;    break;
                case FormatSpec::RGBA32F:     d.format = Format::RGBA32F; break;
                case FormatSpec::SameAsInput: /* keep base.format */      break;
                case FormatSpec::Any:         /* keep base.format */      break;
            }
            if (!d.Valid()) {
                *err = "line " + std::to_string(s.line) + ": '" + s.algoName +
                       "' could not determine output size";
                return false;
            }
            Image img;
            img.Alloc(d);
            s.outputs[p] = Data{std::move(img)};
        }

        RunCtx ctx(in, s.outputs);
        s.algo->RunCPU(ctx);
        s.valid = true;
    }

    return true;
}

} // namespace tglab
