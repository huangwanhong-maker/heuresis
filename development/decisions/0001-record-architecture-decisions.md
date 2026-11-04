# 0001. Record Architecture Decisions

- **Status:** Accepted
- **Date:** 2026-11-04
- **Decided by:** the user and Claude, following the convention shared with the infrastructure and the sibling libraries

## 1. Context

heuresis is long-lived, and every domain that searches for solutions (designs, plans, parameters, programs) will depend on it. Future sessions need to know why its interfaces are as they are, including sessions that do not remember the original discussion.

## 2. Decision

- Record each decision that shapes the structure, interfaces or tooling of heuresis as a numbered record in `development/decisions/`.
- Never rewrite a record. A changed decision gets a new record that supersedes the old one.

## 3. Alternatives Considered

- **Commit messages only.** Rejected, because the reasons become hard to find.
- **A single design document edited in place.** Rejected, because it loses the history of why things changed.

## 4. Consequences

- Each decision costs a small amount of writing.
- The README and the modules' READMEs describe the current state. The records explain how it came about.
