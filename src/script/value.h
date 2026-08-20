// Value: the script interpreter's dynamic value type.
//
// Deliberately small. AlgoHandle is present from M1 so that call-position
// parsing and choose() (M2) need no change to Value's shape.
#pragma once

#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace tglab {

// Reference to one output port of a recorded pipeline stage.
// stage == -1 means a palette (source) image, with port as its index.
struct PortHandle {
    int stage = -1;
    int port  = 0;

    bool operator==(const PortHandle&) const = default;
};

// A registry algorithm name, unapplied. M2: choose() returns one of these.
struct AlgoHandle {
    std::string name;

    // Set by params() to the instance-aware key of the control set declared for
    // this particular use. Empty when the algorithm was not passed through
    // params(). Carried on the value so that calling it applies that set rather
    // than one keyed by algorithm name alone, which two params() calls on the
    // same algorithm would share.
    std::string paramsKey;

    bool operator==(const AlgoHandle&) const = default;
};

// A matrix literal, e.g. [[-1,0,1],[-2,0,2],[-1,0,1]].
struct MatrixValue {
    int                rows = 0;
    int                cols = 0;
    std::vector<double> v;

    double At(int r, int c) const { return v[size_t(r) * size_t(cols) + size_t(c)]; }
    bool operator==(const MatrixValue&) const = default;
};

// A bracketed list of values, e.g. the algorithm list passed to choose().
struct ListValue;

class Value {
public:
    using Storage = std::variant<std::monostate,
                                 double,
                                 std::string,
                                 PortHandle,
                                 AlgoHandle,
                                 MatrixValue,
                                 std::shared_ptr<ListValue>>;

    Value() = default;
    Value(double d)                       : m_v(d) {}
    Value(std::string s)                  : m_v(std::move(s)) {}
    Value(PortHandle p)                   : m_v(p) {}
    Value(AlgoHandle a)                   : m_v(std::move(a)) {}
    Value(MatrixValue m)                  : m_v(std::move(m)) {}
    Value(std::shared_ptr<ListValue> l)   : m_v(std::move(l)) {}

    bool IsNil()    const { return std::holds_alternative<std::monostate>(m_v); }
    bool IsNumber() const { return std::holds_alternative<double>(m_v); }
    bool IsString() const { return std::holds_alternative<std::string>(m_v); }
    bool IsPort()   const { return std::holds_alternative<PortHandle>(m_v); }
    bool IsAlgo()   const { return std::holds_alternative<AlgoHandle>(m_v); }
    bool IsMatrix() const { return std::holds_alternative<MatrixValue>(m_v); }
    bool IsList()   const { return std::holds_alternative<std::shared_ptr<ListValue>>(m_v); }

    double             AsNumber() const { return std::get<double>(m_v); }
    const std::string& AsString() const { return std::get<std::string>(m_v); }
    PortHandle         AsPort()   const { return std::get<PortHandle>(m_v); }
    const AlgoHandle&  AsAlgo()   const { return std::get<AlgoHandle>(m_v); }
    const MatrixValue& AsMatrix() const { return std::get<MatrixValue>(m_v); }
    const ListValue&   AsList()   const { return *std::get<std::shared_ptr<ListValue>>(m_v); }

    const char* TypeName() const;

private:
    Storage m_v;
};

struct ListValue {
    std::vector<Value> items;
};

} // namespace tglab
