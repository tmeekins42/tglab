#include "interp.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "../core/algorithm.h"

namespace tglab {

const char* Value::TypeName() const {
    if (IsNil())    return "nothing";
    if (IsNumber()) return "a number";
    if (IsString()) return "a string";
    if (IsPort())   return "an image";
    if (IsAlgo())   return "an algorithm";
    if (IsMatrix()) return "a matrix";
    if (IsList())   return "a list";
    return "a value";
}

// --- UiState ----------------------------------------------------------------

UiControl* UiState::Find(const std::string& label) {
    for (UiControl& c : m_controls)
        if (c.label == label) return &c;
    return nullptr;
}

UiControl& UiState::FindOrAdd(const UiControl& proto) {
    if (UiControl* existing = Find(proto.label)) {
        // Preserve the user's current value across re-runs, but adopt a
        // changed range/options so editing the script takes effect.
        existing->kind    = proto.kind;
        existing->lo      = proto.lo;
        existing->hi      = proto.hi;
        existing->def     = proto.def;
        existing->options = proto.options;
        if (existing->kind == UiControl::Kind::Slider)
            existing->value = std::clamp(existing->value, proto.lo, proto.hi);
        if (!existing->options.empty())
            existing->selected = std::clamp(existing->selected, 0, int(existing->options.size()) - 1);
        existing->seenThisRun = true;
        return *existing;
    }
    m_controls.push_back(proto);
    m_controls.back().seenThisRun = true;
    return m_controls.back();
}

void UiState::BeginRun() {
    for (UiControl& c : m_controls) c.seenThisRun = false;
}

void UiState::DropUnseen() {
    std::erase_if(m_controls, [](const UiControl& c) { return !c.seenThisRun; });
}

// --- Interpreter ------------------------------------------------------------

namespace {

class Interp {
public:
    Interp(const std::vector<SourceImage>& sources, UiState* ui, Pipeline* out)
        : m_sources(sources), m_ui(ui), m_pipe(out) {}

    bool Run(const Program& prog) {
        for (const Stmt& s : prog.stmts) {
            if (!ExecStmt(s)) return false;
        }
        return true;
    }

    const std::string& Error() const { return m_err; }

private:
    bool Fail(int line, const std::string& msg) {
        if (m_err.empty()) m_err = "line " + std::to_string(line) + ": " + msg;
        return false;
    }

    // `_` is a discard target, not a variable: write-only, like Rust and Go.
    // Only a bare underscore — `_tmp` is an ordinary name.
    static bool IsDiscard(const std::string& name) { return name == "_"; }

    bool ExecStmt(const Stmt& s) {
        // A call may return several ports (multi-output algorithms).
        std::vector<Value> results;
        if (!EvalMulti(*s.value, &results)) return false;

        if (s.targets.empty()) return true;   // expression statement

        // A single target takes the first output, matching what a call already
        // does in single-value position: hysteresis(sobel(x)) uses port 0, so
        // `t = sobel(x)` must mean the same thing. Requiring exact arity here
        // made those two spellings disagree, and made any multi-output
        // algorithm unusable from a choose() dropdown.
        if (s.targets.size() == 1 && results.size() > 1)
            results.resize(1);

        if (s.targets.size() != results.size()) {
            return Fail(s.line, "'" + DescribeCallee(*s.value) + "' returns " +
                                    std::to_string(results.size()) + " value" +
                                    (results.size() == 1 ? "" : "s") + ", " +
                                    std::to_string(s.targets.size()) + " target" +
                                    (s.targets.size() == 1 ? "" : "s") + " given");
        }

        // A single-target assignment from a call remembers which stage it
        // named, so `blur.sigma = x` can later find that stage's parameter.
        if (s.targets.size() == 1 && s.targets[0].object.empty() &&
            !IsDiscard(s.targets[0].field) &&
            s.value->kind == ExprKind::Call && m_lastStage >= 0) {
            m_stageOfVar[s.targets[0].field] = m_lastStage;
        }
        m_lastStage = -1;

        for (size_t i = 0; i < s.targets.size(); ++i) {
            const Target& t = s.targets[i];
            if (t.object.empty() && IsDiscard(t.field)) {
                // `_` discards this output. The algorithm still computed it
                // (outputs are usually fused — sobel's gx/gy/mag come from one
                // loop), and it stays in the stage cache, so adding a viewer
                // for it later costs nothing.
                continue;
            }
            if (t.object.empty()) {
                m_vars[t.field] = results[i];
            } else {
                // `algo.param = value` — assign into a recorded stage's param.
                if (IsDiscard(t.object))
                    return Fail(t.line, "'_' discards a value; it has no parameters to set");
                auto it = m_stageOfVar.find(t.object);
                if (it == m_stageOfVar.end())
                    return Fail(t.line, "'" + t.object + "' is not an algorithm result");
                Stage& st = m_pipe->Stages()[size_t(it->second)];
                ParamBase* p = st.algo->FindParam(t.field);
                if (!p)
                    return Fail(t.line, "'" + st.algoName + "' has no parameter '" + t.field + "'");
                std::string e;
                if (!p->SetFromScript(results[i], &e)) return Fail(t.line, e);
                st.paramHash = st.algo->ParamHash();
            }
        }
        return true;
    }

    static std::string DescribeCallee(const Expr& e) {
        if (e.kind == ExprKind::Call && e.lhs && e.lhs->kind == ExprKind::Ident) return e.lhs->text;
        return "expression";
    }

    // Evaluates an expression that may yield multiple values (a multi-output
    // algorithm call). Everything else yields exactly one.
    bool EvalMulti(const Expr& e, std::vector<Value>* out) {
        if (e.kind == ExprKind::Call) return EvalCall(e, out, /*wantAll=*/true);
        Value v;
        if (!Eval(e, &v)) return false;
        out->push_back(std::move(v));
        return true;
    }

    bool Eval(const Expr& e, Value* out) {
        switch (e.kind) {
            case ExprKind::Number: *out = Value(e.number); return true;
            case ExprKind::String: *out = Value(e.text);   return true;

            case ExprKind::Ident: {
                if (IsDiscard(e.text))
                    return Fail(e.line, "'_' discards a value; it cannot be read back");

                auto it = m_vars.find(e.text);
                if (it != m_vars.end()) { *out = it->second; return true; }
                // A bare name that is not a variable resolves to an algorithm
                // reference, which is what makes choose([a, b, c]) work.
                if (Registry::Get().Contains(e.text)) {
                    *out = Value(AlgoHandle{e.text});
                    return true;
                }
                return Fail(e.line, "unknown name '" + e.text + "'");
            }

            case ExprKind::Unary: {
                Value v;
                if (!Eval(*e.lhs, &v)) return false;
                if (!v.IsNumber()) return Fail(e.line, "cannot negate " + std::string(v.TypeName()));
                *out = Value(-v.AsNumber());
                return true;
            }

            case ExprKind::Binary: {
                Value a, b;
                if (!Eval(*e.lhs, &a) || !Eval(*e.rhs, &b)) return false;
                if (!a.IsNumber() || !b.IsNumber())
                    return Fail(e.line, "arithmetic needs numbers, got " +
                                            std::string(a.IsNumber() ? b.TypeName() : a.TypeName()));
                const double x = a.AsNumber(), y = b.AsNumber();
                double r = 0;
                if (e.text == "+") r = x + y;
                else if (e.text == "-") r = x - y;
                else if (e.text == "*") r = x * y;
                else if (e.text == "/") {
                    if (y == 0) return Fail(e.line, "division by zero");
                    r = x / y;
                } else if (e.text == "%") {
                    if (y == 0) return Fail(e.line, "division by zero");
                    r = std::fmod(x, y);
                }
                *out = Value(r);
                return true;
            }

            case ExprKind::Matrix: {
                MatrixValue m;
                m.rows = int(e.rows.size());
                m.cols = m.rows ? int(e.rows[0].size()) : 0;
                for (const auto& row : e.rows) {
                    if (int(row.size()) != m.cols)
                        return Fail(e.line, "matrix rows must all have the same length");
                    for (const ExprPtr& cell : row) {
                        Value v;
                        if (!Eval(*cell, &v)) return false;
                        if (!v.IsNumber()) return Fail(e.line, "matrix entries must be numbers");
                        m.v.push_back(v.AsNumber());
                    }
                }
                *out = Value(std::move(m));
                return true;
            }

            case ExprKind::List: {
                auto l = std::make_shared<ListValue>();
                for (const ExprPtr& item : e.items) {
                    Value v;
                    if (!Eval(*item, &v)) return false;
                    l->items.push_back(std::move(v));
                }
                *out = Value(std::move(l));
                return true;
            }

            case ExprKind::Member:
                return Fail(e.line, "member access is not supported yet");

            case ExprKind::Call: {
                std::vector<Value> vals;
                if (!EvalCall(e, &vals, /*wantAll=*/false)) return false;
                *out = vals.empty() ? Value() : vals[0];
                return true;
            }
        }
        return Fail(e.line, "unsupported expression");
    }

    // --- builtins -----------------------------------------------------------

    bool EvalCall(const Expr& e, std::vector<Value>* out, bool wantAll) {
        // The callee may be a bare name (builtin or registry algorithm) or any
        // expression evaluating to an algorithm handle.
        std::string calleeName;
        if (e.lhs && e.lhs->kind == ExprKind::Ident) {
            calleeName = e.lhs->text;
            // A variable holding an algorithm shadows the registry name.
            auto it = m_vars.find(calleeName);
            if (it != m_vars.end()) {
                if (!it->second.IsAlgo())
                    return Fail(e.line, "'" + calleeName + "' is " +
                                            std::string(it->second.TypeName()) + ", not something callable");
                calleeName = it->second.AsAlgo().name;
            }
        } else if (e.lhs) {
            Value v;
            if (!Eval(*e.lhs, &v)) return false;
            if (!v.IsAlgo())
                return Fail(e.line, std::string("cannot call ") + v.TypeName());
            calleeName = v.AsAlgo().name;
        }

        if (calleeName == "image")   return CallImage(e, out);
        if (calleeName == "slider")  return CallSlider(e, out);
        if (calleeName == "check")   return CallCheck(e, out);
        if (calleeName == "choose")  return CallChoose(e, out);
        if (calleeName == "display") return CallDisplay(e, out);

        return CallAlgorithm(calleeName, e, out, wantAll);
    }

    // Evaluates arguments once, splitting positional from named.
    struct EvaledArgs {
        std::vector<Value>                            pos;
        std::vector<std::pair<std::string, Value>>    named;
    };

    bool EvalArgs(const Expr& e, EvaledArgs* out) {
        for (const Arg& a : e.args) {
            Value v;
            if (!Eval(*a.value, &v)) return false;
            if (a.name.empty()) out->pos.push_back(std::move(v));
            else                out->named.emplace_back(a.name, std::move(v));
        }
        return true;
    }

    bool CallImage(const Expr& e, std::vector<Value>* out) {
        EvaledArgs a;
        if (!EvalArgs(e, &a)) return false;
        if (a.pos.size() != 1 || !a.pos[0].IsString())
            return Fail(e.line, "image() takes one string: image(\"name\")");
        const std::string& want = a.pos[0].AsString();
        for (const SourceImage& s : m_sources) {
            if (s.name == want) {
                out->push_back(Value(PortHandle{-1, s.index}));
                return true;
            }
        }
        return Fail(e.line, "no image named '" + want + "' in the palette");
    }

    bool CallSlider(const Expr& e, std::vector<Value>* out) {
        EvaledArgs a;
        if (!EvalArgs(e, &a)) return false;
        if (a.pos.size() != 4 || !a.pos[0].IsString() ||
            !a.pos[1].IsNumber() || !a.pos[2].IsNumber() || !a.pos[3].IsNumber())
            return Fail(e.line, "slider() takes (\"label\", min, max, default)");

        UiControl proto;
        proto.kind  = UiControl::Kind::Slider;
        proto.label = a.pos[0].AsString();
        proto.lo    = a.pos[1].AsNumber();
        proto.hi    = a.pos[2].AsNumber();
        proto.def   = a.pos[3].AsNumber();
        proto.value = proto.def;
        if (proto.lo > proto.hi) return Fail(e.line, "slider() min is greater than max");

        UiControl& c = m_ui->FindOrAdd(proto);
        out->push_back(Value(c.value));   // current value, not the default
        return true;
    }

    bool CallCheck(const Expr& e, std::vector<Value>* out) {
        EvaledArgs a;
        if (!EvalArgs(e, &a)) return false;
        if (a.pos.size() != 2 || !a.pos[0].IsString() || !a.pos[1].IsNumber())
            return Fail(e.line, "check() takes (\"label\", default)");

        UiControl proto;
        proto.kind  = UiControl::Kind::Check;
        proto.label = a.pos[0].AsString();
        proto.def   = a.pos[1].AsNumber() != 0 ? 1 : 0;
        proto.value = proto.def;

        UiControl& c = m_ui->FindOrAdd(proto);
        out->push_back(Value(c.value));
        return true;
    }

    // choose("label", [algo_a, algo_b, ...])  — explicit candidates
    // choose("label", "category")             — every algorithm in a category,
    //                                           so a newly written one appears
    //                                           with no script edit.
    //
    // Returns the *selected* algorithm as a callable value, which is what makes
    // `f = choose(...)` followed by `f(img)` work.
    bool CallChoose(const Expr& e, std::vector<Value>* out) {
        EvaledArgs a;
        if (!EvalArgs(e, &a)) return false;
        if (a.pos.size() != 2 || !a.pos[0].IsString())
            return Fail(e.line,
                        "choose() takes (\"label\", [algo, ...]) or (\"label\", \"category\")");

        std::vector<std::string> options;

        if (a.pos[1].IsList()) {
            for (const Value& v : a.pos[1].AsList().items) {
                if (!v.IsAlgo())
                    return Fail(e.line, std::string("choose() list must contain algorithms, found ") +
                                            v.TypeName());
                options.push_back(v.AsAlgo().name);
            }
        } else if (a.pos[1].IsString()) {
            options = Registry::Get().NamesInCategory(a.pos[1].AsString());
            if (options.empty())
                return Fail(e.line, "no algorithms in category '" + a.pos[1].AsString() + "'");
        } else {
            return Fail(e.line, std::string("choose() expects a list or a category name, got ") +
                                    a.pos[1].TypeName());
        }

        if (options.empty()) return Fail(e.line, "choose() needs at least one algorithm");

        UiControl proto;
        proto.kind    = UiControl::Kind::Choose;
        proto.label   = a.pos[0].AsString();
        proto.options = options;

        UiControl& c = m_ui->FindOrAdd(proto);
        const int sel = std::clamp(c.selected, 0, int(options.size()) - 1);
        out->push_back(Value(AlgoHandle{options[size_t(sel)]}));
        return true;
    }

    bool CallDisplay(const Expr& e, std::vector<Value>* out) {
        EvaledArgs a;
        if (!EvalArgs(e, &a)) return false;
        if (a.pos.empty() || !a.pos[0].IsPort())
            return Fail(e.line, "display() takes an image: display(img) or display(img, \"name\")");

        std::string name;
        if (a.pos.size() >= 2) {
            if (!a.pos[1].IsString()) return Fail(e.line, "display()'s second argument must be a name");
            name = a.pos[1].AsString();
        } else {
            name = "View " + std::to_string(m_pipe->Viewers().size() + 1);
        }

        const PortHandle h = a.pos[0].AsPort();
        m_pipe->AddViewer(std::move(name), PortRef{h.stage, h.port});
        (void)out;   // display() yields nothing
        return true;
    }

    bool CallAlgorithm(const std::string& name, const Expr& e,
                       std::vector<Value>* out, bool wantAll) {
        if (name.empty()) return Fail(e.line, "cannot call this expression");
        auto algo = Registry::Get().Create(name);
        if (!algo) return Fail(e.line, "unknown algorithm '" + name + "'");

        EvaledArgs a;
        if (!EvalArgs(e, &a)) return false;

        const PortList inPorts  = algo->Inputs();
        const PortList outPorts = algo->Outputs();

        // Positional arguments bind to input ports, in order.
        if (a.pos.size() != inPorts.size()) {
            return Fail(e.line, "'" + name + "' takes " + std::to_string(inPorts.size()) +
                                    " input" + (inPorts.size() == 1 ? "" : "s") + ", " +
                                    std::to_string(a.pos.size()) + " given");
        }

        std::vector<PortRef> inputs;
        inputs.reserve(a.pos.size());
        for (size_t i = 0; i < a.pos.size(); ++i) {
            if (!a.pos[i].IsPort())
                return Fail(e.line, "'" + name + "' input '" + inPorts[i].name +
                                        "' expects an image, got " + a.pos[i].TypeName());
            const PortHandle h = a.pos[i].AsPort();
            inputs.push_back(PortRef{h.stage, h.port});
        }

        // Named arguments bind to parameters.
        for (const auto& [pname, pval] : a.named) {
            ParamBase* p = algo->FindParam(pname);
            if (!p) return Fail(e.line, "'" + name + "' has no parameter '" + pname + "'");
            std::string perr;
            if (!p->SetFromScript(pval, &perr)) return Fail(e.line, perr);
        }

        const int idx = m_pipe->AddStage(std::move(algo), name, std::move(inputs),
                                         outPorts.size(), e.line);

        // Remember which stage a single-target variable refers to, so that
        // `blur.sigma = x` can find it later.
        m_lastStage = idx;

        const size_t n = wantAll ? outPorts.size() : 1;
        for (size_t p = 0; p < n; ++p)
            out->push_back(Value(PortHandle{idx, int(p)}));
        return true;
    }

    const std::vector<SourceImage>&        m_sources;
    UiState*                               m_ui;
    Pipeline*                              m_pipe;
    std::unordered_map<std::string, Value> m_vars;
    std::unordered_map<std::string, int>   m_stageOfVar;
    int                                    m_lastStage = -1;
    std::string                            m_err;
};

} // namespace

InterpResult Interpret(const Program& prog,
                       const std::vector<SourceImage>& sources,
                       UiState* ui,
                       Pipeline* out) {
    out->Clear();
    ui->BeginRun();

    Interp in(sources, ui, out);
    InterpResult r;
    r.ok = in.Run(prog);
    if (!r.ok) r.error = in.Error();
    else       ui->DropUnseen();
    return r;
}

} // namespace tglab
