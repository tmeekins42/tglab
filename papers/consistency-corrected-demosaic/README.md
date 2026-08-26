# Consistency-Corrected Demosaicing

Accompanies `src/algorithms/demosaic/demosaic_consistent.cpp`.

Bilinear interpolation is a correct low-pass reconstruction --- downsample its
output and its error largely vanishes. What it discards is high-frequency
detail, and that detail is not lost: every sensel is an exact measurement that
bilinear averaged away. The method recovers it, and applies the correction as a
common multiplicative luminance factor so that channel ratios are preserved and
false color cannot be introduced.

Appendix A gives the complete method as pseudocode, in the order the reference
implementation performs it, with the constants it uses.

## Building

    pdflatex paper.tex
    pdflatex paper.tex      # twice, so cross-references resolve

Needs `algpseudocode`, `subcaption`, `booktabs` and `amssymb` --- all standard
in TeX Live, MiKTeX and Overleaf, though MiKTeX may prompt to install them on
first run.

Note the appendix uses `algpseudocode`'s bare `algorithmic` environment rather
than wrapping it in an `algorithm` float. A float cannot break across pages, and
at roughly ninety lines the listing does not fit on one --- wrapped, it ran off
the bottom of the page and stages 5 and 6 were silently absent from the PDF.

Not compiled here --- no LaTeX on the development machine --- so expect to fix
something on the first run. Validated structurally: balanced environments,
matched `\For`/`\EndFor` and `\If`/`\EndIf`, every `\cite` resolving, every
`\ref` resolving, table columns matching the column spec, and every
`\includegraphics` target present.

## Regenerating the data

All from `tools/paper_data.cpp`, registered in CMake. Run from the repository
root.

The table:

    build/Release/paper_data.exe --table <raw>... \
      > papers/consistency-corrected-demosaic/table_main.tex

The figure crops. `x`, `y` and `size` are in source pixels; the origin is
snapped to even coordinates so the CFA phase is preserved, which matters
because an odd offset rotates the Bayer pattern and swaps red with blue:

    build/Release/paper_data.exe --crops <raw> 1200 4700 300 \
      papers/consistency-corrected-demosaic/figures/collar 2.0

    build/Release/paper_data.exe --crops <raw> 1700 2800 320 \
      papers/consistency-corrected-demosaic/figures/hair 2.0

    build/Release/paper_data.exe --crops <raw> 1100 4100 320 \
      papers/consistency-corrected-demosaic/figures/fabric 2.0

The committed figures all came from `_U0A6810.CR3` (Canon EOS R5, ISO 100) at
those coordinates:

- **collar** --- a white fleece collar against a printed red garment. The lead
  figure, and the case that motivated the work: a high-contrast, high-chroma
  boundary crossed by fur finer than the sampling interval. Bilinear fringes
  cyan-green along the whole boundary.
- **hair** --- individual strands against a defocused background.
- **fabric** --- a fine blue-on-red printed pattern near the resolution limit.

They are regenerated from the raw rather than cropped from screenshots, so they
carry no UI chrome and no second round of lossy encoding.

Those coordinates were chosen by rendering the whole frame small with a
coordinate grid every 500 source pixels:

    build/Release/overview.exe <raw> overview.png 2.0

## What it claims, and does not

A novel *combination*, not a new principle.

The residual is Malvar--He--Cutler's (ICASSP 2004), term for term --- arrived
at here independently from a consistency argument, which is some evidence the
reasoning was sound but is not a contribution. The chroma median is Freeman's
(US 4,724,395, 1988), and is standard practice.

What appears to be without direct precedent is diffusing that residual with
edge awareness and injecting it as a common multiplicative luminance factor.
The one structural property worth defending: because the correction is a common
scale factor it preserves channel ratios exactly and therefore cannot introduce
false color. An additive per-channel correction --- including
Malvar--He--Cutler's --- has no such guarantee.

Sections 6 and 7 are deliberate. The negative results (consistency steering
loses to gradient steering 61--72% of the time; the correction saturates past
half strength) and the two methodological failures are the part least likely to
exist anywhere else.

## Authorship

Section 2 discloses the division of work: the method and the visual assessments
are Tim's; the implementation, measurement tools, literature search and drafting
were done with a large language model.

That disclosure is not boilerplate. Section 7 reports two cases where a
plausible quantitative result contradicted the rendered image, and in both the
error was machine-introduced and caught by looking at pictures. A reader should
know which claims carry that risk.

## Known gap

Frame `_MG_9673` at ISO 1000 is an outlier in luminance noise (152% of
bilinear, against 102--122% everywhere else). Reported in the paper rather than
dropped. Not explained.
