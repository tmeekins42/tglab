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
};

struct Program {
    std::vector<Stmt> stmts;
};

} // namespace tglab
