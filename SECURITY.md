# Security policy

Security fixes are made for the latest release.

## Reporting

Report vulnerabilities privately through the repository's **Security** tab ("Report a
vulnerability"). Do not open a public issue. Include the trajclaims version, the operating system,
the impact, and steps that reproduce it.

## Scope

trajclaims downloads two pinned parquet shards and checks their SHA-256. It reads them and writes
under `data/` and `results/`. `trajclaims label` sends sampled agent messages to the local
`claude` CLI with tools, MCP servers, settings files and session saving turned off.
`scripts/fetch_tasks.py` reads task metadata from GitHub at a pinned commit.

In scope:

- writing or deleting files outside `data/` and `results/`;
- code execution triggered by the contents of a dataset row;
- agent text in a dataset row changing what a labeller can do (it has no tools);
- local paths or machine details in committed outputs.

Out of scope: vulnerabilities in dependencies (report those upstream) and statistical
disagreements (open an ordinary issue).
