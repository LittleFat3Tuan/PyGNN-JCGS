# Validation record

Checked on 2026-11-12 on macOS with CPU execution:

| Component | Version |
| --- | --- |
| Python | 3.11.5 |
| PyTorch | 2.4.0 |
| NumPy | 1.26.2 |
| SciPy | 1.11.4 |
| tqdm | 4.66.1 |
| Matplotlib (optional plotting) | 3.8.2 |

## Automated checks

```bash
python -m unittest discover -s tests -v
```

**22 tests pass.** They cover independent scalar definitions of the distribution
losses and masked kernel, double-precision gradient checking,
finite gradients with missing data, tMMD backpropagation, all built-in PRS/QRS
priors, multivariate t scale behavior, validation RNG/buffer/mode preservation,
rank ties and missingness, quantile scaling, remainder batches, paired custom
prior learning, dropout on hidden stages, adaptive updates/stops,
forced continuation, schedule exhaustion/reuse, snapshots, public imports, and
inference-artifact round trips.

The wheel was built without dependency downloads and installed into a temporary
virtual environment with access to the existing runtime dependencies. The suite
also passed against the installed wheel from outside the source checkout. This
checks package contents and import independence; it does not test every supported
dependency version or a fresh network-based dependency resolution.

The editable installation was also checked, and the README quick-start Python
block was executed successfully with both the installed wheel and the editable
package from a temporary working directory.

## End-to-end tutorial runs

These are CPU results with seed 42 and default tutorial settings (512 training
rows, 256 test rows, dimension 2, hidden width 64):

| Tutorial | Epochs | Untrained held-out squared MMD | Trained held-out squared MMD |
| --- | ---: | ---: | ---: |
| Gaussian, fixed bandwidth | 40 | 0.148001 | 0.004904 |
| Clayton, fixed bandwidth | 40 | 0.149334 | 0.006705 |
| Gaussian, adaptive bandwidth | 100 | 0.148001 | 0.002797 |

The adaptive run updated bandwidths at zero-based epoch 53 and reached its
100-epoch cap. Deterministic controller tests separately exercise validation
stopping, forced continuation, and schedule exhaustion. These small runs are
functional checks, not statistical performance comparisons.

The absolute-Gaussian example also ran with `--plot`; the saved loss and sample
figure was inspected. Loading the adaptive inference artifact and drawing 1,024
rank-transformed quasi-random observations through `sample_saved_model` succeeded.
Example outputs were kept in temporary directories outside the repository.

## Limits

CUDA and MPS execution were not available in this session and have not been
validated here. Research-scale runs and reproduction of the article's numerical
experiments were not performed.
