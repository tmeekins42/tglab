# lensfun lens database

`db/*.xml` is the lens database of the lensfun project, copied unmodified from
`data/db/` of https://github.com/lensfun/lensfun at commit
`413bf8f261825f3e3062c17868e7766f2ebe8a21` (2026-10-25).

**Licence: Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)**
— the full text is in [COPYING.CC_BY-SA_3.0](COPYING.CC_BY-SA_3.0), and at
https://creativecommons.org/licenses/by-sa/3.0/.

Attribution: the lensfun database, © the lensfun contributors,
https://lensfun.github.io. The calibrations it holds were contributed by its
many users and maintainers.

The licence covers these data files only. tglab reads them at run time
(`src/core/lensfun.cpp`, its own reader — no lensfun code is used), and the
rest of tglab remains MIT. Any modified copy of these files must be shared
under the same CC BY-SA 3.0 terms.

To update: copy `data/db/*.xml` from a newer lensfun checkout over `db/`, and
change the commit above.
