# MyML (my_tools)

Small, practical Python utilities I use in ML experiments — the kind of “crutches and bicycles” that make notebooks and training scripts more reproducible and convenient.

## What’s inside

### Reproducibility
- `tools.common.set_all_seeds(seed=24)` sets seeds for Python, NumPy, and PyTorch (CPU/GPU) and configures CuDNN for deterministic behavior.

## Installation

This repository is packaged as a small Python module (`my_tools`) containing the `tools/` package.

```bash
pip install -e .
# or
pip install .
```

## Quick start

```python
from tools.common import set_all_seeds

set_all_seeds(42)
```

## Project layout

- `setup.py` — packaging metadata and dependencies
- `tools/common.py` — common ML helpers (currently: seeding)
- `tools/plotting.py` — placeholder for plotting helpers

## Dependencies

Declared in `setup.py`:
- `numpy`
- `pandas`
- `torch`
- `matplotlib`

## Notes

- Deterministic CuDNN can be slower; this is intentional for reproducibility.
- TensorFlow seeding is not implemented yet (placeholder comment in `tools/common.py`).

## Contributing

Issues and PRs are welcome. If you add a helper, please include a short usage example in this README.

## License

No license file is included in this repository yet. If you plan to share or reuse this code, consider adding a license (e.g., MIT/Apache-2.0).
