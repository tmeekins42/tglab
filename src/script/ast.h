#pragma once

#include <memory>
#include <string>
#include <vector>

namespace tglab {

struct Expr;
using ExprPtr = std::unique_ptr<Expr>;

enum class ExprKind {
    Number,
    String,
    Ident,
    Unary,      // -x
    Binary,     // x + y
    Call,       // callee(args...)   — callee is any expr, not just a name
    Member,     // x.y
    Matrix,     // [[..],[..]]
    List,       // [a, b, c]
};

struct Arg {
    std::string name;    // empty when positional
    ExprPtr     value;
};

struct Expr {
    ExprKind kind;
    int      line = 0;
    int      col  = 0;

    // Number
    double number = 0;
    // String / Ident / Member field / Binary or Unary operator
    std::string text;

    // Unary / Binary / Member / Call callee
    ExprPtr lhs;
    ExprPtr rhs;

    // Call
    std::vector<Arg> args;

    // Matrix / List
    std::vector<std::vector<ExprPtr>> rows;   // Matrix
    std::vector<ExprPtr>              items;  // List
};

struct Target {
    std::string object;   // empty for a plain variable
    std::string field;    // variable name, or member name when object is set
    int         line = 0;
};

struct Stmt {
    std::vector<Target> targets;   // empty => expression statement
    ExprPtr             value;
    int                 line = 0;

    // WHICH FILE THIS LINE CAME FROM, empty for the main script.
    //
    // Only meaningful once include() exists, and then it is essential: an
    // included file's statements are spliced into the including program, so a
    // bare "line 12" would name a line in the wrong file. Carried per statement
    // rather than per program because one program legitimately holds
    // statements from several files.
    std::string         file;
};

struct Program {
    std::vector<Stmt> stmts;
};

} // namespace tglab
