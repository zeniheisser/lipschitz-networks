# Lipschitz Networks

Research code and experiments for studying Lipschitz-constrained neural networks
applications in HEP.

## Contents

- `Lipschitz_concept/`: models, training utilities, experiments, notebooks, and plots
- `requirements.txt`: project dependencies
- `requirements-lock.txt`: pinned environment dependencies

The `Tutorials/` directory is intentionally excluded from this repository.

## Setup

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The notebooks in `Lipschitz_concept/` contain the experiments and visualizations.

## CPU use and stored results

Training and floor scans use at most 8 processes or threads. Trained networks and floors are stored in
`Lipschitz_concept/run_cache/`, so running a notebook again only computes what changed. See the top of
`Lipschitz_concept/job_runner.py` for how to change the limit, turn the storage off or clear it.

## Contributing

Issues and pull requests are welcome. Please describe the motivation and
include relevant experiment details when proposing changes.
