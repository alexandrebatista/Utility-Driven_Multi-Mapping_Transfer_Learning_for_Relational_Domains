# Utility-Driven Multi-Mapping Transfer Learning for Relational Domains

This repository contains the implementation of a multi-mapping transfer-learning pipeline for statistical relational learning (SRL). The goal is to improve learning in target domains with limited labeled data by transferring information from one or more source domains through multiple legal vocabulary mappings, while avoiding negative transfer and source-data flooding.

It is the research code for the M.Sc. dissertation *Utility-Driven Multi-Mapping Transfer Learning for Relational Domains* (COPPE/UFRJ) and the paper of the same name.

## 1. Project overview

SRL models such as Relational Dependency Networks (RDNs) are expressive but data-hungry. In relational settings, labeled data is often scarce, and a single source-target pair can admit several valid vocabulary translations. This framework addresses both issues by:

- transferring mapped source instances directly into the target training pool;
- treating each mapping as an independent transfer channel;
- selecting only the most useful instances through a target-guided selector;
- weighting those instances with a generalized focal loss (GFL) formulation;
- preserving the topology of the relational graph by injecting continuous weights into the learning objective instead of pruning structure explicitly.

The intuition is that naively pooling every mapped source instance floods the learner with low-quality data. Scoring instances by *how useful they are to the target task* — rather than by how well they fit the source — lets a single pipeline exploit many mappings at once without being dominated by the bad ones.

## 2. Core idea of the approach

```text
Source Data (Ds)                      Target Data (Dt)
      |                                     |
      | enumerate every type-consistent     |
      | mapping M_DS,j  (Sec. 5.1)          |
      v                                     |
  mapped pool, one channel per mapping      |
      |                                     v
      |                        selector model trained on Dt
      |                        -> AUC-PR becomes the confidence alpha
      v                                     |
      +----------> GFL utility scoring <----+
                          |
                          v
              global ranking, keep top beta
                          |
                          v
          per-channel / per-class normalization
                          |
                          v
          weighted RDN-Boost trained on Dt + survivors
```

More specifically, the framework combines:

- multiple candidate mappings per source-target pair;
- a selector model trained on target data to estimate how consistent a mapped instance is with the target distribution;
- a GFL-based weighting scheme that down-weights easy background examples and emphasizes harder, more informative instances;
- per-channel normalization to avoid a single mapping from dominating the training set.

Every run produces **two matched arms**: the GFL pipeline above, and a `default` baseline that pools *all* mapped source instances unweighted and unfiltered. They share the folds and the round count, so every GFL number has a directly comparable Default number.

## 3. Main features

### 3.1 Instance-based transfer

Instead of revising an entire source model, this approach transfers mapped examples directly into the target domain representation. This makes the pipeline more flexible and avoids expensive model-revision procedures.

### 3.2 Generalized Focal Loss (GFL)

The weighting strategy uses an objective inspired by focal modulation and fairness-based utility functions. For a selector score `c` in (0, 1):

```
l(c)   = (1 - c^alpha) / alpha          # -> -log(c) as alpha -> 0  (cross-entropy)
w      = max( conf * (1 - c)^gamma * l(c) + (1 - conf), 0.01 )
```

- `alpha` controls the shape of the utility curve; the `alpha -> 0` limit recovers cross-entropy;
- `gamma` suppresses easy examples and focuses on hard, target-informative ones;
- `conf` is the target selector's confidence. Both `alpha` and `conf` are set to the selector's validation AUC-PR, so a **weak selector degrades gracefully towards uniform weighting** instead of trusting its own bad scores.

### 3.3 Target-guided selector

A selector model is trained on target data and used to score mapped source instances. These scores are then converted into selection priorities that guide the final transfer pool.

### 3.4 Channel-wise and class-wise normalization

After filtering, surviving examples are normalized by mapping/channel and class so that a high-volume or redundant mapping does not dominate the target learner. The cap is `1 - 0.5 * conf` times the target mass of the same polarity: the more the selector is trusted, the *less* source mass is allowed in.

### 3.5 Continuous weight injection into learning

Instead of physically removing nodes or edges from the relational graph, the framework injects continuous transfer weights directly into the RDN-Boost functional gradient computation. Weights reach the Java learner as a plain `<instance>\t<weight>` file; target instances always carry weight 1.0.

## 4. Repository structure

```text
.
├── graph-embedding/           # main implementation of the experiment and transfer-learning workflow
│   ├── models/                # training, selection, and RDN-Boost workflow code
│   │   ├── transfer_learning_manager.py   # the pipeline (live entry point)
│   │   ├── selector_model.py              # target-guided selector
│   │   ├── model.py                       # weighted RDN-Boost wrapper
│   │   ├── utils.py                       # fold splits + BoostSRL log parsing
│   │   └── selector_models_workflow.py    # earlier selector sweep (legacy, unused)
│   ├── database/              # conversion of data to relational/prolog format
│   ├── mapping/               # generation and application of vocabulary mappings
│   │   ├── mapping.py                     # backtracking search + modes.pl parser
│   │   ├── apply_mappings.py              # rewrites source data under a mapping
│   │   └── utils.py                       # select_mappings() — samples k mappings
│   ├── preprocessed/          # the seven benchmark domains, fold-split
│   ├── dependencies/           # vendored third-party code needed on HTCondor execute nodes
│   │   └── TreeBoostler/                  # tree-based transfer learning for BoostSRL
│   ├── job.py                  # single-mapping job (one mapping id from mappings.json)
│   ├── launcher.sh             # execute-node wrapper for job.py
│   ├── submit_jobs.py          # current HTCondor submitter for job.py — one job per mapping
│   ├── multisource_job.py      # flexible job: explicit --source --target --gamma --size --ids
│   ├── multisource_launcher.sh # execute-node wrapper for multisource_job.py
│   └── submit_experiments.py   # HTCondor submitter for the full k × gamma ablation grid
├── custom-dependencies/       # prebuilt srlearn wheel (outdated, see Sec. 6.1)
├── build_all_srl.sh           # build stock BoostSRL and install srlearn
├── build_all_srl_transfer.sh  # build UtilityMultiDomainRDNBoost and install srlearn
├── init.sh                    # create the Python virtual environment and install dependencies
└── requirements.txt           # Python dependencies of the project
```

`srlearn/` and `UtilityMultiDomainRDNBoost/` are git submodules pointing at this project's forks — run `git submodule update --init --recursive` after cloning. `BoostSRL/`, the stock (unmodified) Java learner built by `build_all_srl.sh`, is **not tracked here** and is not needed unless you specifically want the stock baseline instead of the instrumented `UtilityMultiDomainRDNBoost` learner.

## 5. Main code components

### 5.1 Mapping generation

File: `graph-embedding/mapping/mapping.py`

A source predicate may map to any target predicate of the same arity whose argument types are consistent with the type substitution accumulated so far, or to `null` (the predicate is dropped). `search_tasks_iterative` explores this space by backtracking, enforcing that the type substitution stays a function and that no two source predicates claim the same target predicate. The raw (*coherent*) solutions are filtered twice:

- `filter_solutions` keeps only mappings that produce the **target predicate** — otherwise there is nothing to transfer;
- `filter_robust_solutions` keeps only mappings that *also* produce at least one background predicate sharing a type with the target predicate's arguments — otherwise the transferred examples arrive with no usable relational context.

### 5.2 Mapping application

File: `graph-embedding/mapping/apply_mappings.py`

Source facts that map onto the target predicate are promoted to positive examples; everything else becomes background knowledge. Negatives are then sampled in the target vocabulary at a 2:1 ratio.

Crucially, every constant is rewritten as `<mapping_id>_<constant>` (e.g. `0_david`). This numeric **prefix** prevents instance collision when the channels are pooled: the same source entity seen through two different mappings becomes two distinct constants, so the mappings cannot silently merge into one another.

### 5.3 Transfer-learning manager

File: `graph-embedding/models/transfer_learning_manager.py`

This module orchestrates the full workflow:

- loads the target data and the selected mappings;
- converts source data into a representation compatible with the target;
- trains a selector model on the target domain;
- computes GFL weights for mapped instances;
- reduces the global transfer pool;
- normalizes weights by mapping and class;
- runs the final RDN-Boost training step.

In the code, the central step is `TransferlearningManager._get_transfer_dataset()`, where the pipeline:

1. trains the selector on the target training fold;
2. evaluates the **selector** on a held-out target validation fold — its AUC-PR becomes `alpha` and `conf`;
3. scores every pooled mapped instance with that selector and converts the score into a GFL weight;
4. ranks positives and negatives **globally, across all channels**, and keeps the top `sample_percentage` fraction (this step only runs when more than one mapping is active);
5. rescales the surviving weights per mapping so each channel's source mass stays under the target mass;
6. exports a weight file for the final training step.

`TransferlearningManager.perform_transfer_learning()` runs both arms (GFL and Default) over every fold and saves them into one result file. `set_gamma()` overrides γ before running; `set_mappings()` takes a dict of `{mapping_id: mapping}` and defines which channels are pooled.

### 5.4 RDN-Boost model

File: `graph-embedding/models/model.py`

This component wraps the BoostSRL engine to train a model, test on validation/test data, and retrieve performance metrics. Metrics (AUC ROC, AUC PR, CLL, precision, recall, F1, inference time, learning time) are parsed back out of the BoostSRL logs by `models/utils.py`.

### 5.5 Target selector

File: `graph-embedding/models/selector_model.py`

The selector is a dedicated reliability-estimation component. It receives target data and produces scores indicating how consistent a mapped instance appears to be with the target domain.

### 5.6 Relational database layer

Directory: `graph-embedding/database/`

This part converts raw data into a relational/prolog representation compatible with SRL model training. It is responsible for:

- processing the target domain;
- processing already-mapped source data;
- generating fold splits;
- preparing training and test databases for BoostSRL.

## 6. Execution flow

### 6.1 Environment setup

Prerequisites: Python 3.10 (3.10.19 was used; see `.python-version`) and a JDK (17 was used).

```bash
./init.sh
./build_all_srl_transfer.sh
```

These scripts:

- create a Python virtual environment;
- install the dependencies listed in `requirements.txt`, including the prebuilt wheel in `custom-dependencies/`;
- build the Java learner;
- copy the generated jar into the srlearn directory as `BoostSRL.jar`;
- install srlearn in the Python environment.

Use `build_all_srl.sh` instead when you want the stock learner. A few imports are not pinned in `requirements.txt` and must be added by hand:

```bash
source venv/bin/activate && pip install pandas htcondor
```

(`pandas` for `models/utils.py`, `htcondor` only for the cluster submitters.)

> **The weighted path needs the `srlearn` submodule, not the prebuilt wheel.** `models/model.py` passes `weights=` and `disable_limiters=` to `RDNBoost`. The `srlearn/` submodule (branch `instanceLevelUtility`) declares both keywords, but `custom-dependencies/srlearn-1.0.0-py2.py3-none-any.whl` predates them and raises `TypeError` on the GFL arm. Install from the `srlearn/` submodule before running anything beyond the Default baseline.

### 6.2 Generating the mappings

The mapping-consuming scripts read a `mappings.json` at the working directory (or, for the earlier per-target-file layout, `graph-embedding/mappings/<target_domain>.json`). Both formats key a mapping the same way inside: `predicate_mappings`/`relation_mapping`, `type_mappings`/`term_type_mapping`, `source`, `target`, `id`. Generate them with `mapping.mapping.generate_mappings`:

```python
import json, os
from mapping.mapping import generate_mappings

target, sources = "cora", ["imdb", "yeast", "twitter", "nell_sports", "nell_finances"]
out, next_id = [], 0

for source in sources:
    for mapping in generate_mappings(
        source_domain_name=source, source_dir=f"./preprocessed/{source}/fold00",
        target_domain_name=target, target_dir=f"./preprocessed/{target}/fold00",
    ):
        mapping["id"] = next_id
        mapping["relation_mapping"] = mapping.pop("predicate_mappings")
        mapping["term_type_mapping"] = mapping.pop("type_mappings")
        out.append(mapping)
        next_id += 1

with open("./graph-embedding/mappings.json", "w") as f:
    json.dump(out, f)
```

### 6.3 Local execution

All scripts run from **inside `graph-embedding/`** and resolve `./preprocessed` and `./mappings.json` relative to the working directory.

`multisource_job.py` is the current general-purpose entry point: it pools an explicit, comma-separated list of mapping ids from `mappings.json` for one source → target pair, at a chosen γ:

```bash
cd graph-embedding
python3 multisource_job.py --source imdb --target cora --gamma 0.5 --size 5 --ids "0,1,2,3,4"
```

`--size` is only used to name the output folder (`results/<source>_to_<target>/size_<size>/gamma_<gamma>/ids_<ids>/`) — the pooled set is always exactly the ids passed in `--ids`. Internally it builds `{mapping_id: mapping}` from those ids and calls the same `TransferlearningManager.set_gamma()` / `set_mappings()` / `perform_transfer_learning()` as every other entry point. `target=twitter` is refused (see Section 7 — twitter has too few folds to serve as a target).

A single-mapping run (k = 1) is the `--size 1 --ids "<one id>"` case, or equivalently `job.py <mapping index>` (Section 6.4).

### 6.4 Job-based execution

To run an individual experiment for a single mapping, use `graph-embedding/job.py`:

```bash
cd graph-embedding
python3 job.py 0
```

This script reads `mappings.json` (list format) by index and executes an isolated job with its own results directory, at `results/exp_<idx>_<source>_to_<target>`. If that directory already exists and is non-empty the job **exits immediately**, so a resubmitted batch resumes instead of recomputing. It always runs a single mapping at the manager's default γ (0.5); use `multisource_job.py` to control γ or pool several mappings.

### 6.5 Distributed execution with HTCondor

Two submitters share the same cluster pattern — package a relocatable venv as `venv.tar.gz`, submit via `htcondor.Submit`, and each job resumes on top of an existing non-empty results folder instead of recomputing it:

| Submitter | Executes | Fan-out |
| --- | --- | --- |
| `submit_jobs.py` | `job.py` via `launcher.sh` | one job per mapping id — use this for the single-mapping sweep |
| `submit_experiments.py` | `multisource_job.py` via `multisource_launcher.sh` | one job per (pair, mapping subset of size 1–5, γ ∈ {0.2, 1, 5}) — the full k × γ ablation grid |

Run the current single-mapping sweep with:

```bash
cd graph-embedding
python3 submit_jobs.py
```

or the full ablation grid with:

```bash
cd graph-embedding
python3 submit_experiments.py
```

`submit_experiments.py` groups `mappings.json` by (source, target) pair, skips any pair whose target is `twitter`, takes the first five mapping ids of each remaining pair, and submits one job for **every** combination of subset size (5, 4, 3, 2, 1) × γ (0.2, 1, 5) of those ids — so a pair with 5 mappings produces 31 subsets × 3 γ values = 93 jobs. Results land at `results/<source>_to_<target>/size_<k>/gamma_<g>/ids_<id1-id2-...>/`.

`launcher.sh` / `multisource_launcher.sh` run on the execute node: they unpack a relocatable virtualenv from `venv.tar.gz`, activate it, put the job directory on `PYTHONPATH`, and run the corresponding job script with the arguments HTCondor passed. Each job returns its own `results/` subtree via `transfer_output_files`.

## 7. Supported datasets and scenarios

Seven relational benchmarks ship fold-split under `graph-embedding/preprocessed/`. Each fold directory holds `facts.pl` (background), `pos.pl` / `neg.pl` (labelled examples of the target predicate) and `modes.pl` (the typed vocabulary, in BoostSRL mode syntax). Counts are for `fold00`; `modes.pl` is identical across folds of the same domain.

| Domain          | Folds | Target predicate        | Predicates | Pos | Neg |
| --------------- | ----: | ----------------------- | ---------: | --: | --: |
| `imdb`          |     5 | `workedunder`           |          6 |  55 | 111 |
| `cora`          |     5 | `samevenue`             |         10 | 367 | 687 |
| `uwcse`         |     5 | `advisedby`             |         14 |  34 |  69 |
| `twitter`       |     2 | `accounttype`           |          3 | 125 | 251 |
| `yeast`         |     4 | `proteinclass`          |          7 |  86 | 173 |
| `nell_finances` |     3 | `companyeconomicsector` |         10 | 258 | 517 |
| `nell_sports`   |     3 | `teamplayssport`        |          8 | 131 | 263 |

Any ordered pair of distinct domains is a valid transfer scenario, with one exception: **`twitter` cannot be a target**. `TransferlearningManager` draws a validation fold out of the test folds (`generate_fold_splits(use_validation=True)`), which needs at least 3 folds; `twitter` only has 2. `multisource_job.py` and `submit_experiments.py` both refuse `--target twitter` explicitly for this reason.

`cora`, `imdb` and `uwcse` are trained **without** recursion (`no_recursion_datasets` in `transfer_learning_manager.py`); the rest use it. Fold splits come from `models/utils.py::generate_folds` with `small_dataset=True`: **one fold trains, the rest test**. This is deliberate — the whole premise is low-resource target domains.

## 8. Expected outputs

Everything is written under the `base_path` given to `TransferlearningManager` — under `job.py` and `multisource_job.py` this is a `results/...` subtree keyed by experiment identity (Sections 6.3–6.5):

```text
<base_path>/
├── results/
│   └── <sources>_<target>_<mapping_ids>.json     # numeric suffix appended on collision
├── weights/
│   ├── weights_global_<target>_<fold_id>.txt     # "<instance>\t<weight>", one per line
│   └── sampling_log_fold_<fold_id>_rd_<n>.json   # per-round selection audit
└── selector_models/                              # scratch, deleted after each model
```

`fold_id` encodes the split as `tr_<train folds>_va_<validation folds>`.

Result file:

```jsonc
{
  "target": "cora",
  "source": ["imdb", "imdb", ...],          // one entry per selected mapping
  "mappings": [ /* the full mapping objects */ ],
  "default_transfer_learning_results": [    // baseline arm
    { "fold_configuration": {...}, "train": {...}, "test": {...} }
  ],
  "gfl_transfer_learning_results": [        // GFL arm
    { "transfer_learning": true, "fold_configuration": {...},
      "log": "", "train": {...}, "test": {...} }
  ]
}
```

Each metrics block holds `auc_roc`, `auc_pr`, `cll`, `precision`, `recall`, `f1`, `inference_time_ms`, and — for `train` — `learning_time_s`.

The sampling log records how much data each channel contributed and how much survived the cut. It is the file to read when you want to know *which* mappings the utility function actually kept:

```jsonc
{
  "round": 0,
  "target_samples": { "pos": 55, "neg": 111 },
  "parameters": { "sample_percentage": 0.25 },
  "sources_info": { "<mapping_id>": { "source": "imdb", "pos_before": 812, "neg_before": 1624 } },
  "global_reduction": { "total_pool_pos": 4060, "total_pool_neg": 8120,
                        "selected_pos": 2030, "selected_neg": 4060 }
}
```

## 9. Hyperparameters

| Name | Where | Default | Meaning |
| --- | --- | --- | --- |
| `gamma` (γ) | `TransferlearningManager.gamma`, `set_gamma()` | `0.5` | GFL focusing parameter — how sharply weight shifts toward hard instances. Swept over {0.2, 1, 5} by `submit_experiments.py`. |
| `gfl_lambda` (α) | derived | selector AUC-PR | GFL shape parameter. `α → 0` recovers cross-entropy. |
| `confidence_factor` | derived | selector AUC-PR | Blend between the GFL weight and uniform weighting. |
| `sample_percentage` (β) | `_get_transfer_dataset` | `0.25` | Fraction of the pooled source instances retained after global ranking. Applied only when more than one mapping is active. |
| `max_impact_ratio` | derived | `1 − 0.5 · conf` | Cap on source mass as a multiple of target mass, per polarity. |
| `rounds` | `gfl_transfer_learning`, `default_transfer_learning` | `3` | Repetitions per fold (BoostSRL is stochastic). |
| `size` / number of mapping ids (k) | `--ids` in `multisource_job.py` | caller-supplied | Channels pooled. `1` = single-mapping arm; `submit_experiments.py` sweeps 1–5. |
| `n_estimators` / `max_tree_depth` / `node_size` | `RdnBoost`, `SelectorModel` | `10` / `3` / `2` | RDN-Boost capacity. |
| `neg_pos_ratio` | same | `2` | Negatives sampled per positive. |

`sample_percentage` is not threaded through `gfl_transfer_learning`, so changing the retention fraction currently means editing the default in `_get_transfer_dataset` or passing it at that call site.

## 10. Implementation notes

- The project combines Python and Java components: the main relational learning engine is BoostSRL, while the Python layer handles preprocessing, mapping, selection, and orchestration.
- The transfer strategy is instance-based and uses continuous weights rather than an explicit source-model revision step.
- The architecture is designed to be robust to multiple mappings and to mitigate the negative effects of poorly aligned transfer data.

## 11. Known gaps and caveats

Listed so you do not rediscover them the hard way.

**`models/selector_models_workflow.py` is unused.** It trains/tests a `SelectorModel` per mapping and writes the results back into `mappings.json`. Nothing on the live path (`job.py`, `multisource_job.py`, `models/transfer_learning_manager.py`) calls it; kept for reference in case the offline selector sweep is revisited.

**Scratch directories.** Models write into `<base_path>/selector_models/` and delete themselves afterwards. A crashed run leaves them behind; clear them before rerunning.

## 12. Recommended next steps

- review the data path configuration in `graph-embedding/models/transfer_learning_manager.py`;
- validate the structure of the `preprocessed/`, `mappings.json`, and `results/` folders;
- run a simple experiment (`multisource_job.py` with `--size 1`) to confirm the end-to-end workflow;
- compare results of the baseline without transfer versus the GFL-based pipeline.

## 13. Citation

```bibtex
@mastersthesis{TODO2026utility,
  title  = {Utility-Driven Multi-Mapping Transfer Learning for Relational Domains},
  author = {TODO},
  school = {COPPE/UFRJ},
  year   = {2026}
}
```
