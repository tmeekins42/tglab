#include "parser.h"

#include "lexer.h"

namespace tglab {
namespace {

class Parser {
public:
    Parser(std::vector<Token> toks) : m_toks(std::move(toks)) {}

    bool ParseProgram(Program* out) {
        SkipNewlines();
        while (!At(Tok::End)) {
            Stmt s;
            if (!ParseStmt(&s)) return false;
            out->stmts.push_back(std::move(s));
            SkipNewlines();
        }
        return true;
    }

    const std::string& Error() const { return m_err; }

private:
    // --- token helpers ------------------------------------------------------

    const Token& Cur() const { return m_toks[m_i]; }
    bool At(Tok k) const { return Cur().kind == k; }
    void Advance() { if (m_i + 1 < m_toks.size()) ++m_i; }

    bool Accept(Tok k) {
        if (!At(k)) return false;
        Advance();
        return true;
    }

    bool Expect(Tok k) {
        if (Accept(k)) return true;
        return Fail(std::string("expected ") + TokName(k) + ", got " + Describe(Cur()));
    }

    void SkipNewlines() { while (At(Tok::Newline)) Advance(); }

    static std::string Describe(const Token& t) {
        if (t.kind == Tok::Ident)  return "'" + t.text + "'";
        if (t.kind == Tok::String) return "string";
        if (t.kind == Tok::Number) return "number";
        return TokName(t.kind);
    }

    bool Fail(const std::string& msg) {
        if (m_err.empty())
            m_err = "line " + std::to_string(Cur().line) + ": " + msg;
        return false;
    }

    ExprPtr Make(ExprKind k) {
        auto e = std::make_unique<Expr>();
        e->kind = k;
        e->line = Cur().line;
        e->col  = Cur().col;
        return e;
    }

    // --- statements ---------------------------------------------------------

    // A statement is an assignment if a '=' appears before the end of the line
    // at bracket depth zero. Scanning ahead avoids backtracking the expression
    // parser, and keeps `f(a = 1)` (a named argument) from looking like one.
    bool LooksLikeAssignment() const {
        int depth = 0;
        for (size_t j = m_i; j < m_toks.size(); ++j) {
            switch (m_toks[j].kind) {
                case Tok::LParen: case Tok::LBracket: ++depth; break;
                case Tok::RParen: case Tok::RBracket: --depth; break;
                case Tok::Assign: if (depth == 0) return true; break;
                case Tok::Newline: case Tok::End: return false;
                default: break;
            }
        }
        return false;
    }

    bool ParseStmt(Stmt* out) {
        out->line = Cur().line;

        if (LooksLikeAssignment()) {
            for (;;) {
                Target t;
                t.line = Cur().line;
                if (!At(Tok::Ident)) return Fail("expected a name on the left of '='");
                t.field = Cur().text;
                Advance();
                if (Accept(Tok::Dot)) {
                    if (!At(Tok::Ident)) return Fail("expected a member name after '.'");
                    t.object = t.field;
                    t.field  = Cur().text;
                    Advance();
                }
                out->targets.push_back(std::move(t));
                if (!Accept(Tok::Comma)) break;
            }
            if (!Expect(Tok::Assign)) return false;
        }

        out->value = ParseExpr();
        if (!out->value) return false;

        if (!At(Tok::End) && !At(Tok::Newline))
            return Fail("unexpected " + Describe(Cur()) + " after statement");
        return true;
    }

    // --- expressions --------------------------------------------------------

    ExprPtr ParseExpr() { return ParseAddSub(); }

    ExprPtr ParseAddSub() {
        ExprPtr lhs = ParseMulDiv();
        if (!lhs) return nullptr;
        while (At(Tok::Plus) || At(Tok::Minus)) {
            auto e = Make(ExprKind::Binary);
            e->text = At(Tok::Plus) ? "+" : "-";
            Advance();
            ExprPtr rhs = ParseMulDiv();
            if (!rhs) return nullptr;
            e->lhs = std::move(lhs);
            e->rhs = std::move(rhs);
            lhs = std::move(e);
        }
        return lhs;
    }

    ExprPtr ParseMulDiv() {
        ExprPtr lhs = ParseUnary();
        if (!lhs) return nullptr;
        while (At(Tok::Star) || At(Tok::Slash) || At(Tok::Percent)) {
            auto e = Make(ExprKind::Binary);
            e->text = At(Tok::Star) ? "*" : (At(Tok::Slash) ? "/" : "%");
            Advance();
            ExprPtr rhs = ParseUnary();
            if (!rhs) return nullptr;
            e->lhs = std::move(lhs);
            e->rhs = std::move(rhs);
            lhs = std::move(e);
        }
        return lhs;
    }

    ExprPtr ParseUnary() {
        if (At(Tok::Minus)) {
            auto e = Make(ExprKind::Unary);
            e->text = "-";
            Advance();
            e->lhs = ParseUnary();
            if (!e->lhs) return nullptr;
            return e;
        }
        return ParsePostfix();
    }

    // postfix = primary { "." ident | callSuffix }
    // Calling ANY expression (not just a bare name) is what lets a variable
    // hold an algorithm and be called: ditherFunc(img, ...).
    ExprPtr ParsePostfix() {
        ExprPtr e = ParsePrimary();
        if (!e) return nullptr;

        for (;;) {
            if (At(Tok::Dot)) {
                auto m = Make(ExprKind::Member);
                Advance();
                if (!At(Tok::Ident)) return Fail("expected a member name after '.'"), nullptr;
                m->text = Cur().text;
                Advance();
                m->lhs = std::move(e);
                e = std::move(m);
                continue;
            }
            if (At(Tok::LParen)) {
                auto c = Make(ExprKind::Call);
                Advance();
                if (!ParseArgs(&c->args)) return nullptr;
                if (!Expect(Tok::RParen)) return nullptr;
                c->lhs = std::move(e);
                e = std::move(c);
                continue;
            }
            break;
        }
        return e;
    }

    bool ParseArgs(std::vector<Arg>* out) {
        SkipNewlines();
        if (At(Tok::RParen)) return true;

        for (;;) {
            SkipNewlines();
            Arg a;
            // A named argument is `ident =` — but only when the '=' directly
            // follows, so a bare identifier expression still parses.
            if (At(Tok::Ident) && m_i + 1 < m_toks.size() && m_toks[m_i + 1].kind == Tok::Assign) {
                a.name = Cur().text;
                Advance();
                Advance();
            }
            a.value = ParseExpr();
            if (!a.value) return false;
            out->push_back(std::move(a));

            SkipNewlines();
            if (!Accept(Tok::Comma)) break;
        }
        SkipNewlines();
        return true;
    }

    // A '[' opens either a matrix (rows of numbers) or a flat list. One token
    // of lookahead distinguishes them: '[[' means matrix.
    ExprPtr ParseBracketed() {
        const bool isMatrix = (m_i + 1 < m_toks.size() && m_toks[m_i + 1].kind == Tok::LBracket);
        auto e = Make(isMatrix ? ExprKind::Matrix : ExprKind::List);
        Advance();   // '['
        SkipNewlines();

        if (isMatrix) {
            while (!At(Tok::RBracket)) {
                SkipNewlines();
                if (!Expect(Tok::LBracket)) return nullptr;
                std::vector<ExprPtr> row;
                SkipNewlines();
                while (!At(Tok::RBracket)) {
                    ExprPtr v = ParseExpr();
                    if (!v) return nullptr;
                    row.push_back(std::move(v));
                    SkipNewlines();
                    if (!Accept(Tok::Comma)) break;
                    SkipNewlines();
                }
                if (!Expect(Tok::RBracket)) return nullptr;
                e->rows.push_back(std::move(row));
                SkipNewlines();
                if (!Accept(Tok::Comma)) break;
                SkipNewlines();
            }
        } else {
            while (!At(Tok::RBracket)) {
                SkipNewlines();
                ExprPtr v = ParseExpr();
                if (!v) return nullptr;
                e->items.push_back(std::move(v));
                SkipNewlines();
                if (!Accept(Tok::Comma)) break;
                SkipNewlines();
            }
        }

        SkipNewlines();
        if (!Expect(Tok::RBracket)) return nullptr;
        return e;
    }

    ExprPtr ParsePrimary() {
        if (At(Tok::Number)) {
            auto e = Make(ExprKind::Number);
            e->number = Cur().number;
            Advance();
            return e;
        }
        if (At(Tok::String)) {
            auto e = Make(ExprKind::String);
            e->text = Cur().text;
            Advance();
            return e;
        }
        if (At(Tok::Ident)) {
            auto e = Make(ExprKind::Ident);
            e->text = Cur().text;
            Advance();
            return e;
        }
        if (At(Tok::LBracket)) return ParseBracketed();
        if (At(Tok::LParen)) {
            Advance();
            SkipNewlines();
            ExprPtr e = ParseExpr();
            if (!e) return nullptr;
            SkipNewlines();
            if (!Expect(Tok::RParen)) return nullptr;
            return e;
        }
        Fail("expected a value, got " + Describe(Cur()));
        return nullptr;
    }

    std::vector<Token> m_toks;
    size_t             m_i = 0;
    std::string        m_err;
};

} // namespace

bool Parse(std::string_view src, Program* out, std::string* err) {
    std::vector<Token> toks;
    if (!Lex(src, &toks, err)) return false;

    out->stmts.clear();
    Parser p(std::move(toks));
    if (!p.ParseProgram(out)) {
        *err = p.Error();
        return false;
    }
    return true;
}

} // namespace tglab
