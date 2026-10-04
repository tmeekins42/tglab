// sparse_cholesky — a symmetric positive definite system of small dense
// blocks, stored and factored sparsely.
//
// WHY. Bundle adjustment and global positioning both eliminate their points
// and are left with a reduced CAMERA system, one block per camera. Stored
// densely that is n^2 for n cameras and factored in n^3, which was fine at a
// few hundred cameras and was not at a room scan's 1577: a 9500x9500 matrix,
// 720 MB a copy, factored on every Levenberg-Marquardt step -- fifteen minutes
// of a bundle adjustment. But a camera shares points only with the cameras
// near it in the video and the few it revisits, so almost every block is
// zero. This stores and factors only the blocks that can be non-zero.
//
// HOW. The unknowns come in NODES -- a camera's six or seven parameters, a
// shared focal's one -- and the caller names the node pairs that may couple.
// Analyse() then
//
//   * ORDERS the nodes by minimum degree: eliminating a node couples all of
//     its neighbours to each other (fill), and taking the least connected
//     node each time keeps that small. A video is nearly banded and orders
//     close to its own sequence; a revisit is what makes the choice matter;
//   * records the factor's pattern -- which blocks of L exist, fill
//     included -- so Factor() only does arithmetic, as often as asked.
//
// Factor() is a right-looking block Cholesky over that pattern, and Solve()
// the two triangular solves. The answer is the dense Cholesky's, to rounding:
// only the order of summation differs.
//
// LAYOUT. The caller's VALUES are a flat array holding, for each stored
// block, sizes[i] x sizes[j] doubles, row-major, with i >= j in the caller's
// own node numbering; the diagonal blocks are held in full. Add() takes a
// block in either orientation. The unknowns of node i sit together at
// Scalar(i) in the vectors Solve() reads and writes, in the caller's order.
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace tglab {
namespace linalg {

class BlockSparseCholesky {
public:
    // `sizes[i]` unknowns in node i; `pairs` the node pairs whose block may be
    // non-zero, in either order, repeats and diagonals allowed. False on
    // malformed input (a size under 1, a node out of range).
    bool Analyse(const std::vector<int>& sizes, const std::vector<std::pair<int, int>>& pairs);

    int    Nodes() const { return int(m_size.size()); }
    int    Dim() const { return m_dim; }
    int    Size(int node) const { return m_size[size_t(node)]; }
    int    Scalar(int node) const { return m_scalar[size_t(node)]; }
    size_t NumValues() const { return m_numValues; }
    // Doubles in the factor, fill included: the measure of how well the
    // ordering did, against NumValues() for the matrix itself.
    size_t FactorValues() const { return m_numL; }
    int    Supernodes() const { return int(m_super.size()); }

    // Where block (i, j), i >= j, starts in the values; -1 when the pattern
    // does not have it.
    long long Offset(int i, int j) const;
    // values[block (i, j)] += m, where m is sizes[i] x sizes[j] row-major.
    // Either orientation; for i < j it is stored transposed as block (j, i).
    // A pair outside the pattern is a caller bug and is ignored.
    void Add(double* values, int i, int j, const double* m) const;

    // Offset() for many blocks of one column at once: `rows`, ascending and
    // each >= j, get the offsets of blocks (rows[k], j) in one merge walk,
    // -1 where absent. The elimination loops ask for every pair of cameras
    // a point couples -- thousands for a long track -- and a binary search
    // per pair was what they spent their time on.
    void Offsets(int j, const int* rows, int count, long long* out) const;

    // Factors the matrix in `values`, which is left as it was. False when it
    // is not positive definite.
    bool Factor(const double* values);
    // A x = b in place: `b` in, x out. After a successful Factor().
    void Solve(double* b) const;

private:
    // The caller's pattern, by column: for node j, the rows i >= j present,
    // sorted, each with its offset in the values.
    std::vector<int>       m_size, m_scalar;
    int                    m_dim = 0;
    std::vector<size_t>    m_aColPtr;    // per node j, into the two below
    std::vector<int>       m_aRow;
    std::vector<size_t>    m_aOff;
    size_t                 m_numValues = 0;

    // The elimination order: m_perm[k] is the node eliminated k-th, m_pos its
    // inverse.
    std::vector<int> m_perm, m_pos;

    // THE FACTOR, BY SUPERNODE: a run of consecutive columns (in elimination
    // order) whose patterns nest exactly, stored as one dense row-major panel
    // `width` columns wide. Its top `width` rows are its own nodes -- the
    // dense diagonal block -- and below them come the rows of the nodes it
    // couples to later, m_sRow[rowPtr .. rowPtr + nBelow), each starting at
    // scalar row m_sRowOff of the same index.
    struct Super {
        int    first, last;     // positions [first, last)
        int    width;           // scalars across
        int    rows;            // scalars down, own and below
        size_t panel;           // offset in m_L
        size_t rowPtr;          // into m_sRow / m_sRowOff
        int    nBelow;
    };
    std::vector<Super>  m_super;
    std::vector<int>    m_superOf;   // per position, its supernode
    std::vector<int>    m_colOff;    // per position, its first column within the supernode
    std::vector<int>    m_sRow, m_sRowOff;
    size_t              m_numL = 0;
    std::vector<double> m_L;
    std::vector<int>    m_posScalar;  // scalar offset of position k, in elimination order

    // Scalar row of position `pos` within supernode `s`'s panel.
    int RowIn(int s, int pos) const;

    // Where each of the caller's blocks lands in the factor, in value order:
    // element (r, c) of the block goes to m_L[to + r * ldTo + c], or to
    // m_L[to + c * ldTo + r] when transposed.
    struct Scatter { size_t from, to; int rows, cols, ld; bool transposed; };
    std::vector<Scatter> m_scatter;
};

} // namespace linalg
} // namespace tglab
