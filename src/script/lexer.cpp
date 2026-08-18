#include "lexer.h"

#include <cctype>
#include <cstdlib>

namespace tglab {

const char* TokName(Tok t) {
    switch (t) {
        case Tok::End:      return "end of file";
        case Tok::Number:   return "number";
        case Tok::String:   return "string";
        case Tok::Ident:    return "identifier";
        case Tok::LParen:   return "'('";
        case Tok::RParen:   return "')'";
        case Tok::LBracket: return "'['";
        case Tok::RBracket: return "']'";
        case Tok::Comma:    return "','";
        case Tok::Assign:   return "'='";
        case Tok::Dot:      return "'.'";
        case Tok::Plus:     return "'+'";
        case Tok::Minus:    return "'-'";
        case Tok::Star:     return "'*'";
        case Tok::Slash:    return "'/'";
        case Tok::Percent:  return "'%'";
        case Tok::Newline:  return "end of line";
    }
    return "?";
}

namespace {

bool IsIdentStart(char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_'; }
bool IsIdentCont(char c)  { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

} // namespace

bool Lex(std::string_view src, std::vector<Token>* out, std::string* err) {
    out->clear();

    // Skip a UTF-8 BOM. Windows editors (and PowerShell's Set-Content -Encoding
    // utf8) write one by default, and without this every such script fails on
    // line 1 with a baffling "unexpected character".
    if (src.size() >= 3 && static_cast<unsigned char>(src[0]) == 0xEF &&
        static_cast<unsigned char>(src[1]) == 0xBB &&
        static_cast<unsigned char>(src[2]) == 0xBF) {
        src.remove_prefix(3);
    }

    int line = 1;
    int lineStart = 0;
    size_t i = 0;

    auto col = [&](size_t pos) { return int(pos) - lineStart + 1; };

    auto push = [&](Tok k, size_t pos) {
        Token t;
        t.kind = k;
        t.line = line;
        t.col  = col(pos);
        out->push_back(std::move(t));
    };

    while (i < src.size()) {
        const char c = src[i];

        // Line comments run to end of line.
        if (c == '#') {
            while (i < src.size() && src[i] != '\n') ++i;
            continue;
        }

        if (c == '\n') {
            push(Tok::Newline, i);
            ++i;
            ++line;
            lineStart = int(i);
            continue;
        }

        if (c == ' ' || c == '\t' || c == '\r') { ++i; continue; }

        // Line continuation: a trailing backslash joins the next line, which
        // keeps long nested calls readable.
        if (c == '\\' && i + 1 < src.size() && (src[i + 1] == '\n' || src[i + 1] == '\r')) {
            ++i;
            if (i < src.size() && src[i] == '\r') ++i;
            if (i < src.size() && src[i] == '\n') { ++i; ++line; lineStart = int(i); }
            continue;
        }

        if (std::isdigit(static_cast<unsigned char>(c)) ||
            (c == '.' && i + 1 < src.size() && std::isdigit(static_cast<unsigned char>(src[i + 1])))) {
            const size_t start = i;
            while (i < src.size() && (std::isdigit(static_cast<unsigned char>(src[i])) || src[i] == '.')) ++i;
            // Exponent form (1e-3) — cheap to support, avoids a confusing error.
            if (i < src.size() && (src[i] == 'e' || src[i] == 'E')) {
                size_t save = i;
                ++i;
                if (i < src.size() && (src[i] == '+' || src[i] == '-')) ++i;
                if (i < src.size() && std::isdigit(static_cast<unsigned char>(src[i]))) {
                    while (i < src.size() && std::isdigit(static_cast<unsigned char>(src[i]))) ++i;
                } else {
                    i = save;
                }
            }
            Token t;
            t.kind   = Tok::Number;
            t.number = std::strtod(std::string(src.substr(start, i - start)).c_str(), nullptr);
            t.line   = line;
            t.col    = col(start);
            out->push_back(std::move(t));
            continue;
        }

        if (IsIdentStart(c)) {
            const size_t start = i;
            while (i < src.size() && IsIdentCont(src[i])) ++i;
            Token t;
            t.kind = Tok::Ident;
            t.text = std::string(src.substr(start, i - start));
            t.line = line;
            t.col  = col(start);
            out->push_back(std::move(t));
            continue;
        }

        if (c == '"') {
            const size_t start = i;
            ++i;
            std::string s;
            while (i < src.size() && src[i] != '"') {
                if (src[i] == '\n') break;   // unterminated; report below
                if (src[i] == '\\' && i + 1 < src.size()) {
                    ++i;
                    switch (src[i]) {
                        case 'n':  s += '\n'; break;
                        case 't':  s += '\t'; break;
                        case '\\': s += '\\'; break;
                        case '"':  s += '"';  break;
                        default:   s += src[i]; break;
                    }
                } else {
                    s += src[i];
                }
                ++i;
            }
            if (i >= src.size() || src[i] != '"') {
                *err = "line " + std::to_string(line) + ": unterminated string";
                return false;
            }
            ++i;   // closing quote
            Token t;
            t.kind = Tok::String;
            t.text = std::move(s);
            t.line = line;
            t.col  = col(start);
            out->push_back(std::move(t));
            continue;
        }

        Tok k;
        switch (c) {
            case '(': k = Tok::LParen;   break;
            case ')': k = Tok::RParen;   break;
            case '[': k = Tok::LBracket; break;
            case ']': k = Tok::RBracket; break;
            case ',': k = Tok::Comma;    break;
            case '=': k = Tok::Assign;   break;
            case '.': k = Tok::Dot;      break;
            case '+': k = Tok::Plus;     break;
            case '-': k = Tok::Minus;    break;
            case '*': k = Tok::Star;     break;
            case '/': k = Tok::Slash;    break;
            case '%': k = Tok::Percent;  break;
            default:
                *err = "line " + std::to_string(line) + ": unexpected character '" + std::string(1, c) + "'";
                return false;
        }
        push(k, i);
        ++i;
    }

    push(Tok::End, i);
    return true;
}

} // namespace tglab
