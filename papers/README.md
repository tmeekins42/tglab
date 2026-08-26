# papers/

Technical notes written to accompany algorithms in this repository. None are
submitted anywhere; they exist because writing something up forces a precision
that code comments do not, and because the negative results are usually the
part worth keeping.

## Layout

One directory per paper, named for its subject:

    papers/
      <slug>/
        paper.tex        the document
        README.md        what it covers, and how to regenerate its data
        table_*.tex      generated tables, \input by paper.tex
        figures/         generated figures

Each paper is self-contained: its figures and tables live beside it, so nothing
shares a namespace and a paper can be moved or removed without disturbing the
others.

Generated artifacts are committed rather than gitignored. They are small, and a
paper whose figures cannot be viewed without first building the tool and
locating the raw files is not much use to a reader.

## Current papers

- **consistency-corrected-demosaic** --- `demosaic_consistent`: recovering the
  high-frequency detail bilinear discards, using the sensor's own samples.

## Building

Any of them needs a LaTeX distribution (MiKTeX or TeX Live on Windows), which
is not installed on the development machine. The sources are validated
structurally --- balanced environments, every `\cite` resolving to a `\bibitem`,
every `\ref` to a `\label` --- but have not been compiled, so expect to fix
something on the first run.

    cd papers/<slug>
    pdflatex paper.tex
    pdflatex paper.tex      # twice, so cross-references resolve

## Regenerating data

Every number and figure comes from a tool kept in `tools/`, so that nothing in
a paper is a result whose provenance is "I ran something once". See the
individual paper's README for its own commands.
