#!/usr/bin/env python3
"""
Run Manager -- Job Orchestration and Parameter Sweep
=====================================================
Wraps SU2 runs with parameter sampling, config generation,
job dispatch, result collation, and metadata tracking.

Features:
  - Parameter grids (linspace) or Latin-hypercube sampling
  - Template-based SU2 .cfg generation with parameter substitution
  - Local multiprocessing dispatch (extensible to SLURM)
  - SQLite run registry with metadata (config hash, git commit, mesh ID)
  - Result collation into summary DataFrame

Usage:
  from scripts.orchestration.run_manager import RunManager, ParameterSpace
  space = ParameterSpace()
  space.add_linspace("AOA", 0, 15, 4)
  space.add_choice("TURB_MODEL", ["SA", "SST"])
  mgr = RunManager(template_cfg="naca0012_template.cfg", space=space)
  mgr.generate_configs()
  mgr.run_all(n_workers=4)
  mgr.collate_results()
"""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent


# ===================================================================
# Parameter Space
# ===================================================================
class ParameterSpace:
    """
    Defines the parameter sweep dimensions.

    Supports: linspace grids, choice lists, Latin-hypercube samples.
    """

    def __init__(self):
        self.dimensions: Dict[str, Dict] = {}

    def add_linspace(self, name: str, low: float, high: float, n: int):
        """Add a linearly-spaced parameter."""
        self.dimensions[name] = {
            "type": "linspace",
            "low": low,
            "high": high,
            "n": n,
            "values": np.linspace(low, high, n).tolist(),
        }
        return self

    def add_choice(self, name: str, choices: List[Any]):
        """Add a discrete choice parameter."""
        self.dimensions[name] = {
            "type": "choice",
            "values": choices,
        }
        return self

    def add_values(self, name: str, values: List[float]):
        """Add explicit parameter values."""
        self.dimensions[name] = {
            "type": "explicit",
            "values": values,
        }
        return self

    def add_lhs(self, name: str, low: float, high: float, n: int):
        """
        Add Latin-hypercube sampled parameter.
        Samples are generated when grid is built.
        """
        self.dimensions[name] = {
            "type": "lhs",
            "low": low,
            "high": high,
            "n": n,
        }
        return self

    def build_grid(self, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Build the full parameter grid (Cartesian product of all dimensions).

        For LHS dimensions, samples are generated and treated as explicit values.

        Returns
        -------
        List of parameter dicts, one per configuration.
        """
        # Resolve LHS dimensions first
        resolved = {}
        for name, dim in self.dimensions.items():
            if dim["type"] == "lhs":
                try:
                    from scipy.stats.qmc import LatinHypercube
                    sampler = LatinHypercube(d=1, seed=seed)
                    samples = sampler.random(n=dim["n"])
                    values = (dim["low"] + samples[:, 0] * (dim["high"] - dim["low"])).tolist()
                except ImportError:
                    # Fallback: uniform random
                    rng = np.random.default_rng(seed)
                    values = (dim["low"] + rng.random(dim["n"]) * (dim["high"] - dim["low"])).tolist()
                resolved[name] = values
            else:
                resolved[name] = dim["values"]

        # Cartesian product
        names = list(resolved.keys())
        if not names:
            return [{}]

        import itertools
        all_combos = list(itertools.product(*[resolved[n] for n in names]))
        return [dict(zip(names, combo)) for combo in all_combos]

    def __repr__(self):
        dims = ", ".join(f"{k}({len(v.get('values', []))})" for k, v in self.dimensions.items())
        return f"ParameterSpace({dims})"


# ===================================================================
# Run Configuration
# ===================================================================
@dataclass
class RunConfig:
    """A single simulation configuration."""
    run_id: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    config_file: str = ""
    output_dir: str = ""
    mesh_file: str = ""
    mesh_id: str = ""
    config_hash: str = ""
    git_commit: str = ""
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    wall_time_seconds: float = 0.0

    def __post_init__(self):
        if not self.run_id:
            self.run_id = str(uuid.uuid4())[:8]


def _get_git_commit() -> str:
    """Get current git commit hash, or empty string."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT),
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _hash_config(content: str) -> str:
    """SHA-256 hash of config file content."""
    return hashlib.sha256(content.encode()).hexdigest()[:12]


# ===================================================================
# Run Manager
# ===================================================================
class RunManager:
    """
    Orchestrates SU2 simulation campaigns.

    Parameters
    ----------
    template_cfg : str
        Path to template .cfg file with {PARAM_NAME} placeholders.
    space : ParameterSpace
        Parameter sweep definition.
    base_output_dir : str
        Base directory for all run outputs.
    su2_cmd : str
        SU2 executable command.
    db_path : str
        Path to SQLite database for run tracking.
    """

    def __init__(
        self,
        template_cfg: str,
        space: ParameterSpace,
        base_output_dir: str = "",
        su2_cmd: str = "SU2_CFD",
        db_path: str = "",
    ):
        self.template_cfg = Path(template_cfg)
        self.space = space
        self.base_output_dir = Path(base_output_dir) if base_output_dir else (
            PROJECT / "runs" / "parameter_sweep"
        )
        self.su2_cmd = su2_cmd
        self.db_path = Path(db_path) if db_path else (
            self.base_output_dir / "run_registry.db"
        )
        self.configs: List[RunConfig] = []
        self.git_commit = _get_git_commit()
        self._template_content = ""

    def _init_db(self):
        """Initialize SQLite database."""
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                parameters TEXT,
                config_file TEXT,
                output_dir TEXT,
                mesh_file TEXT,
                mesh_id TEXT,
                config_hash TEXT,
                git_commit TEXT,
                status TEXT,
                start_time TEXT,
                end_time TEXT,
                wall_time_seconds REAL,
                cl REAL,
                cd REAL,
                cm REAL,
                residual_final REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _register_run(self, config: RunConfig):
        """Register a run in the database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, parameters, config_file, output_dir, mesh_file,
                mesh_id, config_hash, git_commit, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (config.run_id, json.dumps(config.parameters), config.config_file,
             config.output_dir, config.mesh_file, config.mesh_id,
             config.config_hash, config.git_commit, config.status),
        )
        conn.commit()
        conn.close()

    def _update_run(self, config: RunConfig, **kwargs):
        """Update run status in the database."""
        conn = sqlite3.connect(str(self.db_path))
        for key, val in kwargs.items():
            conn.execute(f"UPDATE runs SET {key} = ? WHERE run_id = ?",
                         (val, config.run_id))
        conn.commit()
        conn.close()

    def generate_configs(self) -> List[RunConfig]:
        """
        Generate all run configurations from template and parameter space.

        Returns
        -------
        List of RunConfig objects.
        """
        if not self.template_cfg.exists():
            raise FileNotFoundError(f"Template not found: {self.template_cfg}")

        self._template_content = self.template_cfg.read_text()
        grid = self.space.build_grid()

        self._init_db()
        self.configs = []

        for params in grid:
            # Create run ID from parameter values
            param_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
            run_id = hashlib.md5(param_str.encode()).hexdigest()[:8]

            # Substitute parameters in template
            cfg_content = self._template_content
            for key, val in params.items():
                cfg_content = cfg_content.replace(f"{{{key}}}", str(val))
                # Also handle SU2-style: % PARAM = value
                cfg_content = cfg_content.replace(f"__{key}__", str(val))

            # Output directory
            out_dir = self.base_output_dir / f"run_{run_id}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Write config
            cfg_file = out_dir / "config.cfg"
            cfg_file.write_text(cfg_content)

            config = RunConfig(
                run_id=run_id,
                parameters=params,
                config_file=str(cfg_file),
                output_dir=str(out_dir),
                config_hash=_hash_config(cfg_content),
                git_commit=self.git_commit,
                status="ready",
            )

            self._register_run(config)
            self.configs.append(config)

        print(f"Generated {len(self.configs)} configurations")
        return self.configs

    def _run_single(self, config: RunConfig) -> RunConfig:
        """Execute a single SU2 run."""
        config.status = "running"
        config.start_time = datetime.now().isoformat()
        self._update_run(config, status="running", start_time=config.start_time)

        try:
            cmd = [self.su2_cmd, config.config_file]
            t0 = time.time()
            result = subprocess.run(
                cmd, cwd=config.output_dir,
                capture_output=True, text=True,
                timeout=7200,  # 2 hour timeout
            )
            config.wall_time_seconds = time.time() - t0
            config.end_time = datetime.now().isoformat()

            if result.returncode == 0:
                config.status = "completed"
            else:
                config.status = "failed"

            # Save stdout/stderr
            (Path(config.output_dir) / "stdout.txt").write_text(result.stdout)
            (Path(config.output_dir) / "stderr.txt").write_text(result.stderr)

        except subprocess.TimeoutExpired:
            config.status = "timeout"
            config.wall_time_seconds = 7200
        except Exception as e:
            config.status = "failed"
            (Path(config.output_dir) / "error.txt").write_text(str(e))

        config.end_time = datetime.now().isoformat()
        self._update_run(
            config,
            status=config.status,
            end_time=config.end_time,
            wall_time_seconds=config.wall_time_seconds,
        )
        return config

    def run_all(self, n_workers: int = 1):
        """
        Run all configurations.

        Parameters
        ----------
        n_workers : int
            Number of parallel workers. 1 = sequential.
        """
        ready = [c for c in self.configs if c.status in ("ready", "pending")]
        print(f"Running {len(ready)} cases with {n_workers} workers...")

        if n_workers <= 1:
            for i, config in enumerate(ready):
                print(f"  [{i+1}/{len(ready)}] {config.run_id}: {config.parameters}")
                self._run_single(config)
                print(f"    -> {config.status} ({config.wall_time_seconds:.1f}s)")
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(self._run_single, c): c for c in ready}
                for i, future in enumerate(as_completed(futures)):
                    config = futures[future]
                    try:
                        result = future.result()
                        print(f"  [{i+1}/{len(ready)}] {result.run_id}: "
                              f"{result.status} ({result.wall_time_seconds:.1f}s)")
                    except Exception as e:
                        print(f"  [{i+1}/{len(ready)}] {config.run_id}: ERROR ({e})")

    def collate_results(self) -> Dict:
        """
        Collate results from all completed runs.

        Returns
        -------
        dict with summary statistics and per-run results.
        """
        results = {"runs": [], "summary": {}}

        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("SELECT * FROM runs").fetchall()
        columns = [d[0] for d in conn.execute("SELECT * FROM runs").description]
        conn.close()

        for row in rows:
            entry = dict(zip(columns, row))
            if entry.get("parameters"):
                entry["parameters"] = json.loads(entry["parameters"])
            results["runs"].append(entry)

        # Summary
        statuses = [r["status"] for r in results["runs"]]
        results["summary"] = {
            "total": len(results["runs"]),
            "completed": statuses.count("completed"),
            "failed": statuses.count("failed"),
            "pending": statuses.count("pending") + statuses.count("ready"),
            "running": statuses.count("running"),
        }

        # Save
        out_file = self.base_output_dir / "campaign_results.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results collated: {out_file}")
        print(f"  Total: {results['summary']['total']}, "
              f"Completed: {results['summary']['completed']}, "
              f"Failed: {results['summary']['failed']}")

        return results

    def generate_slurm_script(self, partition: str = "compute",
                               walltime: str = "02:00:00",
                               n_tasks: int = 1) -> str:
        """
        Generate SLURM array job submission script.

        Returns
        -------
        Path to the generated script.
        """
        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --time={walltime}",
            f"#SBATCH --ntasks={n_tasks}",
            f"#SBATCH --array=0-{len(self.configs)-1}",
            f"#SBATCH --output={self.base_output_dir}/slurm_%A_%a.out",
            "",
            "# Run directories",
            "RUNS=(",
        ]
        for c in self.configs:
            script_lines.append(f'  "{c.output_dir}"')
        script_lines.extend([
            ")",
            "",
            'RUN_DIR="${RUNS[$SLURM_ARRAY_TASK_ID]}"',
            'cd "$RUN_DIR"',
            f'{self.su2_cmd} config.cfg > stdout.txt 2> stderr.txt',
            "",
        ])

        script_path = self.base_output_dir / "submit_array.sh"
        script_path.write_text("\n".join(script_lines))
        print(f"SLURM script: {script_path}")
        return str(script_path)


# ===================================================================
# CLI
# ===================================================================
def main():
    """Demo: show ParameterSpace capabilities."""
    print("=" * 60)
    print("  RUN MANAGER -- Parameter Sweep Demo")
    print("=" * 60)

    # Example parameter space
    space = ParameterSpace()
    space.add_linspace("AOA", 0, 15, 4)
    space.add_choice("TURB_MODEL", ["SA", "SST"])
    space.add_values("MACH_NUMBER", [0.15])

    grid = space.build_grid()
    print(f"\nParameter space: {space}")
    print(f"Total configurations: {len(grid)}")
    print(f"\nSample configurations:")
    for i, params in enumerate(grid[:5]):
        print(f"  [{i}] {params}")

    # LHS example
    print("\n\nLatin-Hypercube Sampling example:")
    space2 = ParameterSpace()
    space2.add_lhs("AOA", 0, 15, 6)
    space2.add_lhs("MACH_NUMBER", 0.1, 0.3, 6)
    grid2 = space2.build_grid()
    print(f"  LHS samples: {len(grid2)} configurations")
    for i, params in enumerate(grid2[:5]):
        print(f"  [{i}] AOA={params['AOA']:.2f}, M={params['MACH_NUMBER']:.3f}")


if __name__ == "__main__":
    main()
