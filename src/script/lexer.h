#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace tglab {

enum class Tok {
    End,
    Number,
    String,
    Ident,
    LParen, RParen,
    LBracket, RBracket,
    Comma,
    Assign,
    Arrow,      // '=>', the pipe operator
    Dot,
    Plus, Minus, Star, Slash, Percent,
    Newline,
};

struct Token {
    Tok         kind = Tok::End;
    std::string text;      // Ident / String contents
    double      number = 0;
    int         line = 1;
    int         col  = 1;
};

const char* TokName(Tok t);

// Returns false and fills `err` on an unterminated string or bad character.
bool Lex(std::string_view src, std::vector<Token>* out, std::string* err);

} // namespace tglab
