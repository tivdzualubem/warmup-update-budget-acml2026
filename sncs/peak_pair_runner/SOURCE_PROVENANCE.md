# Source provenance

This is new prospective implementation work dated 12 November 2026. It contains no output from the proposed research pair.

The six Python files under `frozen_source/` were copied unchanged from the frozen `warmup-update-budget-acml2026` source within the checked 12 November manuscript revision. Their original source freeze is `bfcbc0bae87d6969fcd4d416cdc457de52c92048`. `FROZEN_SOURCE_HASHES.json` verifies the copied bytes. They supply the scratch data contract, text adapter, medium-4 model, Adam settings, schedule and fingerprint functions. This source identity does not establish byte-for-byte execution provenance for the professor's historical machine.

`reference_utils/` contains the separately versioned corrected evaluator and state/RNG helpers from that same checked revision. These are later analysis/reference additions; they are not claimed to belong to the historical source-freeze commit.

`pair_core.py`, `peak_pair.py`, `check_cpu.py`, the tests and protocol configuration are new files. They replace neither historical code nor historical results. The pair uses new initialization and an explicit shared sampler. Early stopping, validation-best selection and successful-run checkpoint deletion from the old runner are intentionally absent because this proposed test has fixed budgets and retained evidence.

The 8 November feedback explicitly prioritizes a small matched configured-peak first-step comparison. The earlier submission plan and 29 October feedback specify DBpedia, medium-4, seed 42 and the 49,999-step prefix for the larger recovery design. This implementation reuses that base configuration, changes the perturbed first LR to 1e-4 as newly requested, and omits the extra recovery extension. This is an explicit adaptation of the earlier setup, not a claim that November's message repeats every value. The user's direction to proceed supplies authorization; no additional scope-approval request is needed. Full recovery, benchmark expansion and a new guard are not implemented here.

The supplied `ACML_LAUNCHER_HOTFIX_RESUME.zip` is historical execution evidence, not a dependency of this new runner. It is preserved and audited in the separate manuscript package, with SHA-256 `d05ace110f0d702aee6842a81d215a9aa1ffbc9eb279079178a6b46243479a7d`. Its tuple-unwrapping and per-dataset scoping changes explain the errors in the available AG News log. The final applied host files and complete final launcher log remain unavailable.

The local CPU verification record lists every tested source digest and exact library versions. The real-data/device preflight will create its own source/config/data/environment identities. No real-data preflight report or research result is supplied in this package.
