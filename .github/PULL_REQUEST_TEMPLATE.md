## Summary

<!-- One or two sentences. What the PR changes and why. -->

## Substrate discipline checklist

- [ ] No binary float introduced in runtime kernels (`libm4t` / per-query / per-batch paths).
- [ ] No random projections, no random weights anywhere new.
- [ ] No new kernel without a named consumer.
- [ ] Nothing was deleted; superseded code was archived (path:        ).
- [ ] `-Werror` build clean; `ctest` green on the targeted directories.

## Cycle context (if applicable)

<!-- Which LMM cycle does this close, advance, or open? Link the journal files. -->

## Test plan

- [ ] <!-- e.g. ctest --test-dir build passes -->
- [ ] <!-- e.g. specific manual reproduction steps -->

## Risks / follow-ups

<!-- Known gaps, deferred work, things future cycles will need to revisit. -->
