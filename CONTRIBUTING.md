# Contributing

Bug reports should include the command, Python/dependency versions, operating system/compiler, seed, and a minimal failing configuration. Include the exception or a numerical diagnostic such as a KKT/EOS residual. Use GitHub issues after the repository is published.

Before submitting a change:

```bash
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
python reproduce.py check
python reproduce.py plot
```

For sampler or estimator changes, also run `python reproduce.py experiments --quick` and an appropriately sized scientific comparison. The smoke test alone is not evidence that scientific results are preserved. For equation changes, state the paper equation and why the change is necessary; compare both precision definitions when relevant.

Keep existing `data/reference/` files unchanged. New results belong in a new output directory and should include parameters, seeds, convergence checks and a description of any changed protocol. Never remove failed repeats to make a result appear complete. Updating the reference release requires explicitly documenting its provenance and regenerating its checksum manifest.

Python formatting uses Ruff. The scientific core is deliberately small and uses NumPy/SciPy plus local C++17 kernels. Avoid adding mandatory GPU/framework dependencies for unrelated features.
