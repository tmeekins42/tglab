// Reduction: an algorithm that consumes several images along one named axis and
// produces one.
//
// The interface is STREAMING -- Begin, then Accept once per frame, then Finish
// -- rather than "here are all N images". For stills that is a memory
// optimisation: a 60-frame panorama of 45 MP raws is 32 GB if held at once and
// bounded by one frame plus the accumulator if not. For video it is the only
// tractable access pattern, since reaching frame 700 of an inter-frame-coded
// file means decoding from the previous keyframe, and a reduction that asks for
// its inputs in arbitrary order forces either a full decode to disk or repeated
// seeks.
//
// So the interface is settled with video in mind before anything depends on it,
// which is the point of doing it now: a design that only suits stills will not
// stretch, and every merge algorithm written against the wrong shape would have
// to change.
#pragma once

#include <string>

#include "data.h"

namespace tglab {

class CancelToken;

// Implemented by an algorithm that reduces. An algorithm opts in by returning
// true from IsReduction() and overriding these.
//
// The framework calls: Begin once, Accept once per frame in coordinate order,
// then Finish. An accumulator holds one frame's worth of state at a time, which
// is what stops a reduction from asking for the N+1th image under a budget of
// N -- the failure mode the design calls out by name.
class Reducer {
public:
    virtual ~Reducer() = default;

    // Called once before any frame. `count` is how many Accept calls follow,
    // which lets an accumulator size itself, and `axis` names what is being
    // reduced -- a merge may legitimately behave differently over "exposure"
    // than over "position".
    virtual bool Begin(int count, const std::string& axis, std::string* err) = 0;

    // One frame. `index` is its coordinate along the reduced axis, so an
    // algorithm that cares about order (an exposure bracket's stop spacing)
    // has it without the framework having to promise a call order it cannot
    // keep for parallel stills.
    virtual bool Accept(int index, const Image& img, std::string* err) = 0;

    // Produces the result. Called once, after the last Accept.
    virtual bool Finish(Image* out, std::string* err) = 0;
};

} // namespace tglab
