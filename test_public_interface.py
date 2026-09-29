"""Integration checks for release data and command-line failure behavior."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BASE = Path(__file__).resolve().parent


class PublicInterfaceChecks(unittest.TestCase):
    def test_cli_works_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(BASE / "reproduce.py"), "--help"],
                cwd=directory,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("verify-reference", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_invalid_simulation_arguments_create_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid"
            for args in [
                ["--repeats", "0"],
                ["--gap", "0"],
                ["--workers", "0"],
                ["--graphs", "rr", "rr"],
            ]:
                result = subprocess.run(
                    [sys.executable, str(BASE / "run_experiments.py"), "--out", str(output), *args],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(output.exists())

        for script, args in [
            ("run_figure2.py", ["--out"]),
            ("run_experiments.py", ["--out"]),
            ("reproduce.py", ["plot", "--output"]),
        ]:
            result = subprocess.run(
                [sys.executable, str(BASE / script), *args, str(BASE / "data/reference")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("immutable", result.stderr)

    def test_reference_inventory_and_complete_points(self):
        result = subprocess.run(
            [sys.executable, str(BASE / "tools/verify_reference.py")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        reference = BASE / "data/reference"
        summary = json.loads((reference / "experiments/summary.json").read_text())
        self.assertEqual(len(summary), 32)
        self.assertTrue(all(r["repeats"] == 1000 for r in summary))
        self.assertEqual(len(list((reference / "experiments/checkpoints").glob("*.npz"))), 2000)

    def test_partial_results_cannot_be_plotted_as_full_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "summary.json").write_text("[]")
            result = subprocess.run(
                [
                    sys.executable,
                    str(BASE / "plot_experiments.py"),
                    "--experiments",
                    str(output),
                    "--out",
                    str(output / "figures"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(any((output / "figures").glob("*.png")))


if __name__ == "__main__":
    unittest.main()
