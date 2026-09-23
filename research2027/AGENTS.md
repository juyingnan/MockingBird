# Research2027 Agent Guide

This directory is the active NAACL 2027 implementation. It is intentionally
isolated from the historical MockingBird code outside `research2027/`.

## Start here

Before doing work:

1. Pull the current branch.
2. Read `PLAN.md`.
3. Read `STATUS.md`.
4. Check recent commits on the working branch.
5. If the task depends on prior experiments, inspect `RUNS.md` and the relevant
   artifacts rather than relying on chat/session memory.

Current project state is defined by `STATUS.md`, not by an agent's recollection.

## Execution discipline

- Execute one milestone or clearly bounded task at a time.
- Use PowerShell as the primary shell; do not require Bash.
- Prefer synthetic or targeted tests before expensive runs.
- Reuse validated embedding caches; do not recompute them without a reason.
- Keep model extraction and large experiment expansion behind the gates in
  `PLAN.md`.
- Do not browse the web or spawn subagents during engineering work unless
  explicitly requested.

## Multi-agent and multi-machine coordination

Git is the coordination layer.

When more than one person, agent, or compute environment may be active:

- use task-specific branches;
- pull before starting;
- do not edit the same branch concurrently;
- commit reproducible code/config changes before handing work off;
- push the branch before another worker continues.

A handoff should be understandable from the repository alone.

After substantive work:

1. Commit code/config changes.
2. Update `STATUS.md` if milestone or project state changed.
3. Append substantial experiment executions to `RUNS.md`.
4. Push the branch.

Do not create long narrative agent logs. Git records code history;
`STATUS.md` records current state; `RUNS.md` records experiment provenance.

## Experiment provenance

For every substantial run, record enough in `RUNS.md` for another worker to
understand and reproduce it:

- date;
- owner/agent;
- git commit and branch;
- milestone / experiment ID;
- research question or hypothesis;
- dataset and split;
- model/checkpoint;
- config and command;
- seeds;
- compute environment at a reproducible level;
- output/artifact location;
- completion status;
- short result summary;
- anomalies or failures.

Use high-level compute labels such as:

- `local / RTX 5080`
- `IU HPC / <GPU type>`
- `Microsoft-approved hackathon Azure / <GPU type>`

Never record credentials, tokens, subscription/tenant IDs, internal endpoints,
or other sensitive infrastructure details.

## Scientific discipline

Keep these three claims separate:

1. **Encoding** — is non-target information accessible from the representation?
2. **Reliance** — does the already-trained emotion classifier depend on it?
3. **Generalization consequence** — does removing access to it change learning
   under the corresponding distribution shift?

Do not treat probe accuracy as evidence of reliance.
Do not treat reliance as evidence of harmful generalization.
Do not call an attribute a shortcut unless the corresponding evidence supports
that interpretation.

Null, negative, and contradictory results must be preserved.

## Paper synchronization

The paper repository is:

`https://github.com/chenyueg/NAACL2027_emotion`

Experimental execution and raw scientific facts live here.

Only scientifically interpretable, traceable results should cross into the paper
repository's `RESULTS_LEDGER.md`. Every transferred result should point back to
an experiment artifact plus a Git commit or experiment ID in this repository.

A newly produced number is not automatically a paper claim.
