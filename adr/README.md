# Architecture decision records

This directory records *why* a significant technical decision was made, for whoever
maintains this code later (including future us). It's maintainer-facing, not part of the
user docs under `docs/` -- it's deliberately not in `mkdocs.yml`'s nav, and `docs_dir` is
`docs`, so mkdocs never scans this directory at all.

## When to write one

Not every decision needs one. Write a record when someone maintaining this code later
could reasonably ask "why didn't we just do the obvious/simpler thing instead?" and the
answer isn't visible from the code or a docstring alone -- e.g. a tradeoff that was
measured, an alternative that was tried and rejected, a constraint that shaped the design.
Routine implementation choices, anything already explained by a code comment or docstring,
and anything that would only matter to a user of the library (that belongs in `docs/`
instead) don't need one here.

## Format

One file per decision: `NNNN-short-slug.md`, numbered sequentially (doesn't need to be
gapless). Each file:

```markdown
# NNNN. Title

Status: Proposed | Accepted | Deprecated | Superseded by NNNN
Date: YYYY-MM-DD

## Context

What problem or constraint forced a decision. The forces at play -- what made the obvious
approach not good enough.

## Decision

What was actually chosen, concretely enough that someone could verify it against the
current code.

## Consequences

What this costs (new dependencies, complexity, constraints on future changes) and what it
buys (the thing the decision was for). Good and bad consequences both belong here, not
just the ones that justify the decision.
```

## Rule: records are immutable once accepted

Don't edit an old record to reflect a later change of mind -- write a new one, set its
`Status` to explain what it replaces, and set the old record's `Status` to
`Superseded by NNNN`. The point is a real history: what we believed and why, at the time,
even after we later learned better. If a decision is abandoned before anything is built on
it, `Status: Deprecated` with a short note is fine instead of a new record.
