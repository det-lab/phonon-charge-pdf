# 0004. Ppq_compiled rejects data outside its region

Status: Proposed
Date: 2026-11-09

## Context

`Ppq_compiled(ep_range, eq_range, ep_data, eq_data, ...)` normalizes each band's
density over the box `ep_range × eq_range`, so its normalized densities are a
probability density on that box only. An event outside the box isn't in the
model's sample space at all. Its "normalized density" is a finite number with
no meaning, and summing its log into a likelihood biases the fit with no
visible sign.

`__init__` stores the data as given and never compares it with the box. This
is easy to get wrong:

- the published tables fix the region (2.5–350 / 0.75–200 keV), so a user's
  existing cuts (e.g. Joseph's 2–200 / 1–100 keV, a different box entirely) are
  likely not to match it;
- the docs' examples pass `ep_data, eq_data` as "your measured events" and only
  say in comments that they must already be cut to the region;
- an `MCMC_demo/model.py`-style loader has to remember to cut, every time.

## Decision

`Ppq_compiled.__init__` checks that every event satisfies
`ep_min <= ep <= ep_max` and `eq_min <= eq <= eq_max`, and that the data are
finite. Otherwise it raises `ValueError` reporting how many events fail, on
which side, and the extreme values, with a one-line hint for cutting them
(`inside = (ep >= ...) & ...`).

It never cuts the data itself. Silently dropping events would change the dataset
behind the user's back, and the count of events is part of the analysis.

## Consequences

- An unmatched cut becomes an immediate, specific error.
- Closed vs open edges: events exactly on the boundary are accepted (the density
  is defined there). Loaders that cut with strict inequalities stay valid.
- It costs one vectorized comparison at construction, nothing per call.
- Verification: a test that a single event just outside each of the four edges
  raises, and that events on the edges don't.
