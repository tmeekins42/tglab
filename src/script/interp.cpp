#include "interp.h"

#include <algorithm>
#include <cmath>
#include <map>
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
        // A control still sitting on its old default follows a changed one.
        //
        // Defaults are fixed for almost every parameter, but kelvin's comes
        // from the image -- and on startup the control is created before the
        // raw finishes loading, so it is built at 0 and then preserved here
        // forever. The slider showed 0 K on a file whose metadata says 5381.
        //
        // Only when untouched: a value the user moved, or the script set, is
        // still theirs to keep.
        if (existing->kind == UiControl::Kind::Slider &&
            existing->value == existing->def && existing->def != proto.def) {
            existing->value = proto.def;
        }
        existing->def     = proto.def;
        existing->display = proto.display;
        existing->group   = proto.group;
        existing->help    = proto.help;
        existing->step    = proto.step;
        existing->softLo  = proto.softLo;
        existing->softHi  = proto.softHi;
        existing->options = proto.options;
        existing->defaultIndex = proto.defaultIndex;
        if (existing->kind == UiControl::Kind::Slider)
            existing->value = std::clamp(existing->value, proto.lo, proto.hi);
        if (!existing->options.empty())
            existing->selected = std::clamp(existing->selected, 0, int(existing->options.size()) - 1);
        existing->seenThisRun = true;
        existing->declOrder   = m_declOrder++;
        return *existing;
    }
    m_controls.push_back(proto);
    m_controls.back().seenThisRun = true;
    m_controls.back().declOrder   = m_declOrder++;
    return m_controls.back();
}

void UiState::BeginRun() {
    for (UiControl& c : m_controls) c.seenThisRun = false;
    m_declOrder = 0;
}

void UiState::DropUnseen() {
    std::erase_if(m_controls, [](const UiControl& c) { return !c.seenThisRun; });

    // Reorder to match this run's declarations.
    //
    // FindOrAdd() appends new controls but leaves existing ones in place, so
    // the vector holds first-*ever*-seen order rather than the script's. That
    // is invisible until a control is re-declared: switching one algorithm in a
    // two-instance script drops its controls and re-adds them at the end, which
    // silently swaps the two groups in the panel -- so the slider labelled A
    // sits under B's header and adjusting it appears to do nothing.
    //
    // stable_sort so controls sharing a declaration index (which cannot happen
    // today, but would if one call ever declared several) keep their relative
    // order.
    std::stable_sort(m_controls.begin(), m_controls.end(),
                     [](const UiControl& a, const UiControl& b) {
                         return a.declOrder < b.declOrder;
                     });
}

// --- Interpreter ------------------------------------------------------------

namespace {

class Interp {
public:
    Interp(const std::vector<SourceImage>& sources, UiState* ui, Pipeline* out,
           std::string defaultDemosaic)
        : m_sources(sources), m_ui(ui), m_pipe(out),
          m_defaultDemosaic(std::move(defaultDemosaic)) {}

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

        // Which params() declaration applies to this call, if any. It travels
        // on the algorithm value rather than being looked up by name, so
        // params(op, "A") and params(op, "B") stay distinct.
        std::string paramsKey;

        if (e.lhs && e.lhs->kind == ExprKind::Ident) {
            calleeName = e.lhs->text;
            // A variable holding an algorithm shadows the registry name.
            auto it = m_vars.find(calleeName);
            if (it != m_vars.end()) {
                if (!it->second.IsAlgo())
                    return Fail(e.line, "'" + calleeName + "' is " +
                                            std::string(it->second.TypeName()) + ", not something callable");
                paramsKey  = it->second.AsAlgo().paramsKey;
                calleeName = it->second.AsAlgo().name;
            }
        } else if (e.lhs) {
            Value v;
            if (!Eval(*e.lhs, &v)) return false;
            if (!v.IsAlgo())
                return Fail(e.line, std::string("cannot call ") + v.TypeName());
            paramsKey  = v.AsAlgo().paramsKey;
            calleeName = v.AsAlgo().name;
        }

        if (calleeName == "image")   return CallImage(e, out);
        if (calleeName == "mosaic")  return CallMosaic(e, out);
        if (calleeName == "slider")  return CallSlider(e, out);
        if (calleeName == "check")   return CallCheck(e, out);
        if (calleeName == "choose")  return CallChoose(e, out);
        if (calleeName == "params")  return CallParams(e, out);
        if (calleeName == "display") return CallDisplay(e, out);

        return CallAlgorithm(calleeName, paramsKey, e, out, wantAll);
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
            if (s.name != want) continue;
            // A group of raws is not handled yet. The demosaic image() would
            // normally insert takes one image, and mapping it across a set is
            // the broadcasting step that has not been built -- so say so here
            // rather than handing the script undemosaiced sensor data that
            // would merge into a plausible-looking mess.
            if (!s.shape.IsScalar()) {
                if (s.isMosaic)
                    return Fail(e.line, "'" + want + "' is a group of raw images, which "
                                        "cannot be demosaiced yet");
                out->push_back(Value(PortHandle{-1, s.index}));
                return true;
            }
            // An ordinary image is handed straight through.
            if (!s.isMosaic) {
                out->push_back(Value(PortHandle{-1, s.index}));
                return true;
            }

            // A mosaic needs demosaicing before anything else can use it, and
            // the script must not have to say so. Insert the default method as
            // an ordinary stage, so it caches, runs on the GPU, and shows up in
            // the stage list like everything else.
            //
            // Recorded once per source: two image("x") calls in one script
            // should share a stage rather than demosaicing twice.
            if (auto it = m_demosaicStage.find(s.index); it != m_demosaicStage.end()) {
                out->push_back(Value(PortHandle{it->second, 0}));
                return true;
            }

            // Repair stuck sensels before demosaicing.
            //
            // Before, because a demosaic smears one bad sample across a
            // neighbourhood: afterwards it is no longer a single-pixel outlier
            // and the repair would have to guess at a blob whose shape depends
            // on which demosaic ran. Tim's 5D has seven such sensels, at the
            // same coordinates in every frame.
            //
            // Inserted automatically for the same reason the demosaic is: a
            // script should not have to know the sensor has defects. It is an
            // ordinary registered stage, so it appears in the stage list, has
            // its own sliders, and can be turned off -- which matters for
            // astrophotography, where a star IS a genuine one-pixel highlight.
            PortRef demosaicInput{-1, s.index};
            if (auto repair = Registry::Get().Create("hot_pixel_repair")) {
                const int rs = m_pipe->AddStage(std::move(repair), "hot_pixel_repair",
                                                {PortRef{-1, s.index}}, 1, e.line);
                demosaicInput = PortRef{rs, 0};
            }

            auto algo = Registry::Get().Create(m_defaultDemosaic);
            if (!algo)
                return Fail(e.line, "unknown default demosaic '" + m_defaultDemosaic + "'");

            const int stage = m_pipe->AddStage(std::move(algo), m_defaultDemosaic,
                                               {demosaicInput}, 1, e.line);
            m_demosaicStage[s.index] = stage;
            out->push_back(Value(PortHandle{stage, 0}));
            return true;
        }
        return Fail(e.line, "no image named '" + want + "' in the palette");
    }

    // mosaic("name") -- the undemosaiced sensor data.
    //
    // The counterpart to image()'s automatic demosaic: this is what a script
    // uses to demosaic explicitly, which is the only way to compare two methods
    // side by side. Errors on a non-raw source rather than returning something
    // that merely looks like a mosaic.
    bool CallMosaic(const Expr& e, std::vector<Value>* out) {
        EvaledArgs a;
        if (!EvalArgs(e, &a)) return false;
        if (a.pos.size() != 1 || !a.pos[0].IsString())
            return Fail(e.line, "mosaic() takes one string: mosaic(\"name\")");

        const std::string& want = a.pos[0].AsString();
        for (const SourceImage& s : m_sources) {
            if (s.name != want) continue;
            if (!s.isMosaic)
                return Fail(e.line, "'" + want + "' is not a raw image, so it has "
                                    "no sensor mosaic; use image(\"" + want + "\")");
            out->push_back(Value(PortHandle{-1, s.index}));
            return true;
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
        if (a.pos.size() < 2 || a.pos.size() > 3 || !a.pos[0].IsString())
            return Fail(e.line,
                        "choose() takes (\"label\", [algo, ...]) or (\"label\", "
                        "\"category\"), with an optional third argument naming "
                        "the default");

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

        // An optional third argument names which option starts selected.
        //
        // Without it the default is whichever algorithm sorts first in the registry,
        // which is alphabetical and therefore arbitrary -- "demosaic_bilinear"
        // beats "demosaic_passthrough" by accident of spelling. The script
        // should be able to say what it means, and the dropdown still overrides
        // it at run time.
        if (a.pos.size() == 3) {
            std::string want;
            if (a.pos[2].IsString())      want = a.pos[2].AsString();
            else if (a.pos[2].IsAlgo())   want = a.pos[2].AsAlgo().name;
            else return Fail(e.line, std::string("choose()'s default must be an "
                                                 "algorithm or its name, got ") +
                                         a.pos[2].TypeName());

            const auto it = std::find(options.begin(), options.end(), want);
            if (it == options.end())
                return Fail(e.line, "'" + want + "' is not one of the options for '" +
                                        proto.label + "'");
            proto.defaultIndex = int(it - options.begin());
            proto.selected     = proto.defaultIndex;
        }

        UiControl& c = m_ui->FindOrAdd(proto);
        const int sel = std::clamp(c.selected, 0, int(options.size()) - 1);
        out->push_back(Value(AlgoHandle{options[size_t(sel)]}));
        return true;
    }

    // params(algo) declares a UI control for every parameter the algorithm
    // has, using its own name, range and default, and returns the algorithm so
    // the call reads naturally:
    //
    //     op  = choose("operator", "threshold")
    //     out = params(op)(src)
    //
    // Switching the dropdown swaps the whole control set, because controls the
    // run does not re-declare are dropped (UiState::DropUnseen). A newly
    // written algorithm exposes its parameters with no script edit at all --
    // the same win choose() gives for algorithm selection.
    bool CallParams(const Expr& e, std::vector<Value>* out) {
        EvaledArgs a;
        if (!EvalArgs(e, &a)) return false;
        if (a.pos.empty() || a.pos.size() > 2 || !a.pos[0].IsAlgo())
            return Fail(e.line,
                        "params() takes an algorithm and an optional name: "
                        "params(op) or params(op, \"A\")");
        if (a.pos.size() == 2 && !a.pos[1].IsString())
            return Fail(e.line, "params()'s second argument must be a name");

        const std::string& name = a.pos[0].AsAlgo().name;
        auto probe = Registry::Get().Create(name);
        if (!probe) return Fail(e.line, "unknown algorithm '" + name + "'");

        // The optional instance name is what lets the same algorithm appear
        // twice with independent settings. Without it, params(op) twice on one
        // algorithm shares a single set of controls -- so a script comparing an
        // algorithm against itself could not vary anything.
        //
        // Computed here rather than further down because the control lookup
        // below needs it: a control's label carries the instance.
        const std::string instance = a.pos.size() == 2 ? a.pos[1].AsString() : std::string();
        const std::string key      = instance.empty() ? name : name + "@" + instance;

        // params() takes options of its own, distinct from the algorithm's
        // parameters.
        //
        // `auto_exposure` is one: it does not adjust the image, it decides
        // WHERE the adjustments start. That is a property of the call, not a
        // control -- as a checkbox it read as something you could toggle to see
        // the effect, and it could not honestly behave that way, because by the
        // time a control has a value the defaults have already been chosen.
        //
        // So the script says it, and every slider is then an ordinary control
        // the user can override.
        bool wantAuto = false;

        for (const auto& [pname, pval] : a.named) {
            if (pname == "auto_exposure") {
                if (!pval.IsNumber())
                    return Fail(e.line, "params()'s auto_exposure expects 0 or 1");
                wantAuto = pval.AsNumber() != 0.0;
                continue;
            }
            // Anything else is one of the algorithm's own parameters, applied
            // to the probe so a script-set value is what the control opens at.
            ParamBase* p = probe->FindParam(pname);
            if (!p) return Fail(e.line, "'" + name + "' has no parameter '" + pname + "'");
            std::string perr;
            if (!p->SetFromScript(pval, &perr)) return Fail(e.line, perr);
        }

        // Let the algorithm derive defaults from the source before its controls
        // are described -- basic_adjust opens its kelvin slider at whatever
        // temperature the camera chose, rather than at a sentinel.
        //
        // params() names an algorithm, not a call, so there is no single source
        // to point at; the first raw one is used. In practice a develop script
        // has exactly one, and a control's default is a starting point rather
        // than a claim about a specific image.
        for (const SourceImage& s : m_sources) {
            if (!s.isMosaic) continue;
            SourceFacts facts;
            facts.isMosaic     = true;
            facts.asShotKelvin = s.asShotKelvin;
            facts.asShotTint   = s.asShotTint;
            // Only when the script asked. The measurement is always available;
            // whether it steers the defaults is the script's decision.
            facts.hasExposure  = wantAuto && s.hasAutoExposure;
            facts.autoExposure = s.autoExposure;
            facts.autoHighlights = s.autoHighlights;
            facts.autoShadows  = s.autoShadows;
            facts.autoBlacks   = s.autoBlacks;
            probe->PrepareDefaults(facts);
            break;
        }

        // The group is what the user sees, so it leads with the instance name
        // when there is one: "A (bilateral)" reads better than the reverse.
        const std::string group =
            instance.empty() ? name : instance + " (" + name + ")";

        ParamValues pv;
        for (ParamBase* p : probe->Params()) {
            UiControl proto;
            // Unique across the panel; never shown.
            proto.label   = key + "." + p->Name();
            // Shown inside the group box, where the algorithm name is already
            // in the header and repeating it is what clipped the panel.
            proto.display = p->Name();
            proto.group   = group;
            if (const char* h = p->Help()) proto.help = h;
            if (!p->DescribeControl(&proto)) continue;
            UiControl& c = m_ui->FindOrAdd(proto);
            pv.values.emplace_back(p->Name(), c.value);
        }

        // Keyed by the same instance-aware key, so two params() calls on one
        // algorithm no longer overwrite each other's pending values.
        m_pendingParams[key] = std::move(pv);

        // The algorithm value carries the key forward, so the call site knows
        // which pending set belongs to it.
        AlgoHandle h = a.pos[0].AsAlgo();
        h.paramsKey = key;
        out->push_back(Value(h));
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

    bool CallAlgorithm(const std::string& name, const std::string& paramsKey,
                       const Expr& e, std::vector<Value>* out, bool wantAll) {
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
            const PortRef ref{h.stage, h.port};

            // Shape check. Scalar is the default, so this fires only when
            // something has actually produced a set and handed it to an
            // algorithm that takes one image.
            const Shape got = ShapeAt(ref);
            if (inPorts[i].shape == ShapeSpec::Scalar && !got.IsScalar()) {
                return Fail(e.line, "'" + name + "' input '" + inPorts[i].name +
                                        "' takes a single image, got " + got.ToString() +
                                        " -- reduce it first, e.g. over=\"" +
                                        got.Axes()[0].name + "\"");
            }
            inputs.push_back(ref);
        }

        // Values from a params() declaration are applied first, so an explicit
        // named argument in the same call still wins.
        // Looked up by the key params() attached to this algorithm value, so
        // two params() calls on one algorithm apply their own settings.
        if (auto pit = m_pendingParams.find(paramsKey.empty() ? name : paramsKey);
            pit != m_pendingParams.end()) {
            for (const auto& [pname, pvalue] : pit->second.values) {
                if (ParamBase* p = algo->FindParam(pname)) {
                    std::string perr;
                    p->SetFromScript(Value(pvalue), &perr);
                }
            }
        }

        // Named arguments bind to parameters, except over=, which the framework
        // owns: the axis decides the SHAPE of the result, so it has to be known
        // while the pipeline is built rather than when the stage runs.
        std::string overAxis;
        for (const auto& [pname, pval] : a.named) {
            if (pname == "over") {
                if (!pval.IsString())
                    return Fail(e.line, "over= takes an axis name, e.g. over=\"exposure\"");
                overAxis = pval.AsString();
                continue;
            }
            ParamBase* p = algo->FindParam(pname);
            if (!p) return Fail(e.line, "'" + name + "' has no parameter '" + pname + "'");
            std::string perr;
            if (!p->SetFromScript(pval, &perr)) return Fail(e.line, perr);
        }

        // First input's shape decides what SameAsInput/Reduced mean. Captured
        // before the move, since AddStage takes `inputs` by value.
        const Shape inShape = inputs.empty() ? Shape::Scalar() : ShapeAt(inputs[0]);

        // over= is only meaningful on a reduction, and a reduction needs one.
        const bool isReduction = algo->IsReduction();
        int reduceIdx = -1;
        if (!overAxis.empty()) {
            if (!isReduction)
                return Fail(e.line, "'" + name + "' is not a reduction, so it takes no over=");
            reduceIdx = inShape.Find(overAxis);
            if (reduceIdx < 0) {
                return Fail(e.line, "'" + name + "' cannot reduce over '" + overAxis +
                                        "': its input is " + inShape.ToString());
            }
        } else if (isReduction) {
            // One axis is unambiguous, so naming it is optional. More than one
            // is a real choice and the script has to make it.
            if (inShape.Rank() == 1) {
                reduceIdx = 0;
                overAxis  = inShape.Axes()[0].name;
            } else if (inShape.Rank() > 1) {
                return Fail(e.line, "'" + name + "' needs over= to say which axis of " +
                                        inShape.ToString() + " to reduce");
            } else {
                return Fail(e.line, "'" + name + "' reduces a set, but its input is a single image");
            }
        }

        const int idx = m_pipe->AddStage(std::move(algo), name, std::move(inputs),
                                         outPorts.size(), e.line, overAxis);

        // Propagate shapes to this stage's outputs so a later line can be
        // checked against them.
        for (size_t p = 0; p < outPorts.size(); ++p) {
            Shape s;   // Scalar and Any both produce one image today
            if (outPorts[p].shape == ShapeSpec::SameAsInput) s = inShape;
            else if (outPorts[p].shape == ShapeSpec::Reduced && reduceIdx >= 0)
                s = inShape.Without(reduceIdx);
            if (!s.IsScalar()) m_shapeOf[{idx, int(p)}] = s;
        }

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
    // Values declared by params(), applied when that algorithm is next called.
    // Keyed by algorithm name.
    struct ParamValues { std::vector<std::pair<std::string, double>> values; };
    std::unordered_map<std::string, ParamValues> m_pendingParams;

    // The demosaic image() inserts for a mosaic source, and the stage already
    // recorded for each one -- two image("x") calls should share a stage rather
    // than demosaicing the same sensor data twice.
    std::string                    m_defaultDemosaic;
    std::unordered_map<int, int>   m_demosaicStage;   // palette index -> stage

    // Shape of every port produced so far, keyed by {stage, port}.
    //
    // Build-time, symbolic: the interpreter never sees a Data, so the only way
    // to check a shape mismatch at the line that caused it is to propagate
    // shapes forward as stages are recorded. A stage's outputs follow its
    // declared ShapeSpec; a palette image's shape is whatever its entry holds.
    std::map<std::pair<int, int>, Shape> m_shapeOf;

    Shape ShapeAt(const PortRef& r) const {
        if (r.stage < 0) {
            // Palette image: scalar for a single dropped file, which is every
            // entry today.
            for (const SourceImage& s : m_sources)
                if (s.index == r.port) return s.shape;
            return Shape::Scalar();
        }
        auto it = m_shapeOf.find({r.stage, r.port});
        return it == m_shapeOf.end() ? Shape::Scalar() : it->second;
    }

    std::string                            m_err;
};

} // namespace

InterpResult Interpret(const Program& prog,
                       const std::vector<SourceImage>& sources,
                       UiState* ui,
                       Pipeline* out,
                       const std::string& defaultDemosaic) {
    out->Clear();
    ui->BeginRun();

    Interp in(sources, ui, out, defaultDemosaic);
    InterpResult r;
    r.ok = in.Run(prog);
    if (!r.ok) r.error = in.Error();
    else       ui->DropUnseen();
    return r;
}

} // namespace tglab
