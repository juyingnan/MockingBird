# AGENTS.md

This repository contains legacy MockingBird code plus the isolated NAACL 2027
research implementation.

## Scope

- Treat `classification/`, `clustering/`, `utils/`, and `visualization/`
  as legacy code. Do not modify them for NAACL 2027 work.
- Put all new NAACL 2027 implementation, experiments, and research artifacts
  under `research2027/`.
- Before working under `research2027/`, read:
  1. `research2027/PLAN.md`
  2. `research2027/STATUS.md`
  3. `research2027/AGENTS.md`

## Working rules

- Git is the shared source of truth across people, agents, and compute
  environments. Do not rely on old chat/session state.
- Work on task-specific branches when another worker may be active.
- Do not have two agents modify the same branch concurrently.
- Prefer the smallest experiment or test that answers the current question.
- Do not expand scope, run expensive sweeps, browse the web, or spawn subagents
  unless the task explicitly requires it.
- Keep secrets, credentials, subscription/tenant IDs, and private infrastructure
  details out of the repository.

The detailed research workflow, run logging, and paper synchronization rules live
in `research2027/AGENTS.md`.
