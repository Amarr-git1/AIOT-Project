# Chapter 3: Requirements and Specifications

This chapter describes the resources, data, and system capabilities required
to run and evaluate the network-security project. The project has two related
but separate workflows: a synthetic network simulation for evaluating
defensive policies, and offline CSV-based workflows for training or evaluating
flow classifiers. Requirements for the large dataset apply to those
data-oriented workflows, not to the main simulation by itself.

## 3.1 Hardware Requirements

The project does not define a mandatory hardware specification. The
requirements below are practical recommendations based on the implemented
workflows and the size of the dataset currently in the repository.

| Component | Recommended specification | Purpose |
|---|---|---|
| Processor | 64-bit, 4 or more CPU cores | Runs Python, graph operations, model training, and evaluation. |
| Memory | 8 GB minimum for the simulator; 16 GB or more recommended for CSV training and full evaluation | The CSV is approximately 685 MB, and data processing and model training require additional working memory. |
| Storage | At least 2 GB free for the dataset and basic outputs; 10 GB or more recommended for repeated evaluations | The dataset, checkpoints, logs, charts, and timestamped evaluation outputs can accumulate. |
| GPU | Optional; a CUDA-capable NVIDIA GPU can accelerate supported PyTorch workloads | The main simulation and evaluation can run on CPU; a GPU is not required. |
| Display | Standard display capable of showing a terminal and PNG/PDF charts | Used to run commands and inspect generated reports. |

Actual run time and memory use depend on the selected dataset sample size,
training epochs, batch size, number of policy episodes, and whether a GPU is
available. Full evaluation of the approximately 2.2-million-row CSV may take
substantially longer than a short simulation or smoke-sized experiment.

## 3.2 Software Requirements

The project is implemented in Python. Its scripts use the following software
and libraries:

| Software or library | Role |
|---|---|
| Python 3 | Runs the simulation, model training, evaluation scripts, and tests. The recorded evaluation environment used Python 3.14.0. |
| PyTorch | Provides tensors, neural-network modules, optimization, and checkpoint handling. |
| PyTorch Geometric | Provides graph convolution and graph attention layers used by the GCN/GAT components. Its version must be compatible with the installed PyTorch build and CPU/CUDA configuration. |
| NumPy | Supports feature arrays, numerical calculations, and simulation state. |
| pandas | Reads and processes CSV datasets and writes metrics and reports. |
| scikit-learn | Provides preprocessing, data splitting, and classification metrics. |
| Matplotlib and Seaborn | Generate evaluation and comparison charts. |
| openpyxl | Optional dependency for Excel workbook export. CSV export remains available without it. |
| Python standard library | Supports command-line arguments, file operations, random number generation, JSON, and unit tests. |

The repository does not include a root `requirements.txt`, dependency lock
file, or environment specification. Install compatible versions of the
libraries required by the workflow being run. For example, the main
simulation needs PyTorch, PyTorch Geometric, and NumPy; the full evaluation
also uses pandas, scikit-learn, and plotting libraries. Use the
PyTorch/PyTorch Geometric installation instructions appropriate to the
machine's operating system and CPU or CUDA setup.

The repository is developed in a Windows environment, and commands in its
guides are shown for execution from the project root. The Python scripts use
standard Python libraries and can be run on other supported operating
systems when dependencies and file paths are configured appropriately.

## 3.3 Dataset Requirements

### 3.3.1 Dataset Description

The CSV data file used by the project's dataset workflows is `combine.csv`.
It contains network-flow records with a ground-truth `Label` column. Each
record represents a flow described by numeric traffic and protocol
measurements; the input file does not provide host endpoint or timestamp
columns for reconstructing a live network graph.

The dataset profile recorded for the current file reports:

- **2,214,469 total rows**, of which **2,214,468 have a label**.
- **79 columns**: one label column and **78 feature columns**.
- A file size of approximately **685 MB** (684,702,235 bytes).
- Labels for benign traffic and multiple attack families.

The project uses the dataset in two distinct ways. `train_gcn.py` uses CSV
records to construct synthetic graph samples for training the eight-feature
network GCN. `run_model_evaluation.py` separately trains and evaluates a
binary flow detector using numeric CSV columns. That detector's target is
**BENIGN versus ATTACK**; it is not a multiclass attack-family classifier.
The evaluation runner also conducts separate policy experiments in the
synthetic simulator. The CSV records are not replayed as traffic in the main
simulation.

### 3.3.2 Dataset Features

The CSV provides 78 input features and one target label. The features
describe flow behavior and protocol statistics, including:

| Feature group | Examples |
|---|---|
| Flow and packet counts | Flow duration, forward and backward packet totals, and subflow packet counts |
| Packet lengths and bytes | Total forward/backward bytes, packet-length minimum, maximum, mean, standard deviation, and variance |
| Flow rates and timing | Flow bytes per second, flow packets per second, and flow or directional inter-arrival-time statistics |
| TCP and protocol indicators | FIN, SYN, RST, PSH, ACK, URG, CWE, and ECE flag counts; header lengths; initial window-byte values |
| Activity and idle behavior | Active and idle duration statistics, including means, standard deviations, minima, and maxima |
| Directional and service indicators | Destination port, down/up ratio, average packet size, and forward/backward segment statistics |

The `Label` column is the ground truth and must not be used as a model input.
For binary detection, `BENIGN` is the benign class and non-benign labels are
grouped as `ATTACK`. The evaluation workflow also reports selected attack
families separately using one-versus-benign subsets; this reporting does not
change the detector into a multiclass model.

These CSV fields are different from the eight per-node features used by the
main simulation. Those node features are compromise status, risk score,
isolation status, vulnerability score, flow duration, packet rate, byte
rate, and normalized node degree. The main simulation initializes its
50-node feature table in code rather than loading `combine.csv`.

### 3.3.3 Dataset Characteristics

The recorded profile for the current CSV provides the following important
characteristics:

- **Imbalanced labels:** 1,672,837 records are BENIGN and 541,631 are
  ATTACK, corresponding to approximately 75.54% benign and 24.46% attack
  records among labeled rows.
- **Multiple attack families:** The profile includes DDoS, PortScan, Bot,
  Infiltration, several denial-of-service labels, and Heartbleed. The
  family counts vary considerably; for example, the recorded counts include
  231,073 DoS Hulk records, 128,027 DDoS records, 158,930 PortScan records,
  and 1,966 Bot records.
- **Missing and non-finite values:** The recorded profile contains 1,215
  missing cells, including one missing label, and 3,735 infinite numeric
  cells. The evaluation workflow converts non-finite values to missing
  values and applies median imputation and standard scaling, with
  preprocessing fitted on training records.
- **Repeated and conflicting feature rows:** The profile reports repeated
  feature hashes and 691 feature groups with conflicting labels. The
  evaluation workflow groups rows by a hash of the input features, excluding
  the label, when preparing splits. This helps keep identical feature
  records together, but conflicting labels remain a limitation of the
  dataset.
- **No graph endpoints or timestamps:** The dataset profile reports no
  timestamp or host-endpoint columns. The flow detector therefore treats
  rows as individual samples and does not infer network edges from the CSV.

These values describe the dataset profile recorded by a previous evaluation
run. If `combine.csv` is replaced or modified, regenerate the profile before
using these figures as current dataset statistics.

## 3.4 Functional Requirements

The system shall provide the following capabilities:

1. **Construct a simulated network:** Create the 50-node Fat Tree-style
   topology, its adjacency relationships, and the node-level feature
   representation.
2. **Estimate node risk:** Pass node features and graph connectivity to a
   GCN and produce a risk estimate for each node.
3. **Represent the agent state:** Combine the eight features and one GCN
   prediction for each of the 50 nodes into the state used by the SAC agent.
4. **Select defensive actions:** Produce one SAC action value per node and
   map it to one of eight responses: allow traffic, monitor, raise an alert,
   rate-limit traffic, block a source IP, block a destination port, isolate
   a host, or terminate a session.
5. **Simulate attack and defense transitions:** Apply synthetic attack
   behavior and model how defensive actions and isolation affect compromise
   and propagation.
6. **Calculate rewards and learn:** Score transitions using changes in
   compromise and risk, isolation outcomes, and action costs; store
   transitions and update the SAC policy when training is enabled.
7. **Explain decisions and export results:** Generate heuristic,
   human-readable node explanations and save selected simulation outputs
   as CSV and, when `openpyxl` is available, Excel files.
8. **Train and evaluate graph models:** Provide a workflow to train the
   eight-feature GCN using CSV-derived synthetic graph samples.
9. **Evaluate flow detection:** Read the CSV, preprocess numeric features,
   train a binary BENIGN-versus-ATTACK detector, and calculate classification
   metrics.
10. **Evaluate defense policies:** Train and compare implemented policies
    in the synthetic simulator, including the supported generic, DDoS,
    PortScan, and Bot scenarios.
11. **Save evaluation artifacts:** Write metrics, traces, plots, model
    checkpoints, and run metadata to the configured output directories so
    that completed runs can be inspected.

Requirements 8–10 describe separate workflows; they do not mean that the
main simulation consumes CSV traffic or that the detector directly controls
the simulated or a real network.

## 3.5 Non-Functional Requirements

1. **Usability:** Provide command-line entry points and clear outputs so
   users can run simulations, tests, training, and evaluation from the
   repository root.
2. **Performance:** Support CPU execution and process the large CSV in
   chunks in the evaluation workflow. Training duration and resource use
   should be adjustable through the available run parameters.
3. **Reliability:** Validate required dataset columns, handle missing and
   non-finite numeric values during evaluation, and save results in
   dedicated run directories rather than silently replacing previous
   evaluations.
4. **Reproducibility:** Allow evaluation workflows to use configured random
   seeds and record dataset, split, and run metadata. Results from the main
   simulation can still vary because attack propagation uses randomness;
   seeding NumPy and PyTorch alone does not seed every random source.
5. **Maintainability:** Keep topology generation, graph models, agent
   decisions, attack behavior, reward calculation, explainability, and
   evaluation in separable modules.
6. **Interpretability:** Associate decisions with readable explanations and
   feature-contribution summaries. These explanations are heuristic and
   should not be presented as causal model interpretations.
7. **Portability:** Run on CPU-capable Python environments, with GPU
   acceleration optional and dependent on compatible PyTorch and
   PyTorch Geometric installations.
8. **Data integrity:** Keep labels separate from model inputs, retain
   preprocessing and split metadata for dataset evaluations, and document
   label imbalance and duplicate/conflicting feature records when
   interpreting results.
9. **Safety and scope:** Restrict attack and response behavior to the
   simulator. The project does not connect to live networks, inspect live
   packets, or enforce security controls on real devices.

## 3.6 Summary

The project requires a Python environment with compatible PyTorch and
PyTorch Geometric installations, along with the numerical, data-processing,
and plotting libraries required by each workflow. A standard CPU is
sufficient for the main simulation; more memory, storage, and optionally a
GPU are recommended for full-scale CSV experiments.

The current dataset is a large, labeled flow table with 78 numeric input
features and one label column. It is imbalanced, contains missing and
non-finite values, and does not contain the endpoint or timestamp
information needed to construct a graph directly. Dataset-based detection
and training are therefore kept distinct from the main synthetic network
simulation. Together, the functional and non-functional requirements define
a controlled, explainable, and reproducible research environment while
clearly limiting claims to simulated behavior and offline evaluation.
