# Utility-Driven Multi-Mapping Transfer Learning for Relational Domains

This repository contains the implementation of a multi-mapping transfer-learning pipeline for statistical relational learning (SRL). The goal is to improve learning in target domains with limited labeled data by transferring information from one or more source domains through multiple legal vocabulary mappings, while avoiding negative transfer and source-data flooding.

## 1. Project overview

SRL models such as Relational Dependency Networks (RDNs) are expressive but data-hungry. In relational settings, labeled data is often scarce, and a single source-target pair can admit several valid vocabulary translations. This framework addresses both issues by:

- transferring mapped source instances directly into the target training pool;
- treating each mapping as an independent transfer channel;
- selecting only the most useful instances through a target-guided selector;
- weighting those instances with a generalized focal loss (GFL) formulation;
- preserving the topology of the relational graph by injecting continuous weights into the learning objective instead of pruning structure explicitly.

## 2. Core idea of the approach

The pipeline is organized into three main stages:

```text
Source Data (Ds)
  -> apply mapping M_DS,j
  -> build a mapped pool for each channel
  -> target-guided filtering and weighting
  -> train weighted RDN-Boost model on the target domain
```

More specifically, the framework combines:

- multiple candidate mappings per source-target pair;
- a selector model trained on target data to estimate how consistent a mapped instance is with the target distribution;
- a GFL-based weighting scheme that down-weights easy background examples and emphasizes harder, more informative instances;
- per-channel normalization to avoid a single mapping from dominating the training set.

## 3. Main features

### 3.1 Instance-based transfer

Instead of revising an entire source model, this approach transfers mapped examples directly into the target domain representation. This makes the pipeline more flexible and avoids expensive model-revision procedures.

### 3.2 Generalized Focal Loss (GFL)

The weighting strategy uses an objective inspired by focal modulation and fairness-based utility functions. In practice:

- $\alpha$ controls the shape of the utility curve;
- $\gamma$ suppresses easy examples and focuses on hard, target-informative ones;
- the target selector provides a confidence signal used to compute instance-level priorities.

### 3.3 Target-guided selector

A selector model is trained on target data and used to score mapped source instances. These scores are then converted into selection priorities that guide the final transfer pool.

### 3.4 Channel-wise and class-wise normalization

After filtering, surviving examples are normalized by mapping/channel and class so that a high-volume or redundant mapping does not dominate the target learner.

### 3.5 Continuous weight injection into learning

Instead of physically removing nodes or edges from the relational graph, the framework injects continuous transfer weights directly into the RDN-Boost functional gradient computation.

## 4. Repository structure

```text
.
├── BoostSRL/                  # Java BoostSRL engine used by the pipeline
├── srlearn/                   # Python interface for the SRL engine
├── graph-embedding/           # main implementation of the experiment and transfer-learning workflow
│   ├── models/                # training, selection, and RDN-Boost workflow code
│   ├── database/              # conversion of data to relational/prolog format
│   ├── mapping/               # application of vocabulary mappings
│   ├── main.py                # local execution entry point
│   ├── job.py                 # execution of individual jobs
│   └── htcondor.py            # HTCondor integration for distributed execution
├── build_all_srl.sh           # build BoostSRL and install srlearn
├── init.sh                    # create the Python virtual environment and install dependencies
└── requirements.txt           # Python dependencies of the project
```

## 5. Main code components

### 5.1 Transfer-learning manager

Main file: graph-embedding/models/transfer_learning_manager.py

This module orchestrates the full workflow:

- loads the target data and the selected mappings;
- converts source data into a representation compatible with the target;
- trains a selector model on the target domain;
- computes GFL weights for mapped instances;
- reduces the global transfer pool;
- normalizes weights by mapping and class;
- runs the final RDN-Boost training step.

In the code, the central step is implemented in `TransferlearningManager._get_transfer_dataset()`, where the pipeline:

1. trains the selector on the target training fold;
2. evaluates mapped instances on the target validation fold;
3. computes a GFL-based weight;
4. applies global pool reduction;
5. normalizes the example volume per mapping;
6. exports a weight file for the final training step.

### 5.2 RDN-Boost model

File: graph-embedding/models/model.py

This component wraps the BoostSRL engine to:

- train a model;
- test on validation/test data;
- retrieve performance metrics.

### 5.3 Target selector

File: graph-embedding/models/selector_model.py

The selector is a dedicated reliability-estimation component. It receives target data and produces scores indicating how consistent a mapped instance appears to be with the target domain.

### 5.4 Relational database layer

Directory: graph-embedding/database/

This part converts raw data into a relational/prolog representation compatible with SRL model training. It is responsible for:

- processing the target domain;
- processing already-mapped source data;
- generating fold splits;
- preparing training and test databases for BoostSRL.

## 6. Execution flow

### 6.1 Environment setup

```bash
./init.sh
./build_all_srl.sh
```

These scripts:

- create a Python virtual environment;
- install the dependencies listed in requirements.txt;
- build the Java BoostSRL component;
- copy the generated jar into the srlearn directory;
- install srlearn in the Python environment.

### 6.2 Local execution

The current entry point of the project is graph-embedding/main.py. Execution is organized around a model workflow and a mapping workflow.

Example usage:

```bash
cd graph-embedding
python3 main.py
```

### 6.3 Job-based execution

To run an individual experiment for a given mapping, use graph-embedding/job.py:

```bash
cd graph-embedding
python3 job.py 0
```

This script reads a mapping configuration file and executes an isolated job with its own results directory.

### 6.4 Distributed execution with HTCondor

The graph-embedding/htcondor.py module prepares submissions to an HTCondor cluster. It creates a set of jobs, each with its own results and log folder.

## 7. Supported datasets and scenarios

The repository was designed to evaluate relational transfer scenarios between domains such as:

- IMDB ↔ Cora
- IMDB ↔ UW-CSE
- Twitter → Yeast
- NELL Finances ↔ NELL Sports

## 8. Expected outputs

A run typically generates:

- result folders in graph-embedding/results/ or in experiment-specific subfolders;
- JSON files with experiment metrics and configuration;
- weight files for transferred examples;
- training and execution logs.

## 9. Implementation notes

- The project combines Python and Java components: the main relational learning engine is BoostSRL, while the Python layer handles preprocessing, mapping, selection, and orchestration.
- The transfer strategy is instance-based and uses continuous weights rather than an explicit source-model revision step.
- The architecture is designed to be robust to multiple mappings and to mitigate the negative effects of poorly aligned transfer data.

## 10. Recommended next steps

- review the data path configuration in graph-embedding/main.py and graph-embedding/models/transfer_learning_manager.py;
- validate the structure of the preprocessed/, mappings/, and results/ folders;
- run a simple experiment to confirm the end-to-end workflow;
- compare results of the baseline without transfer versus the GFL-based pipeline.
