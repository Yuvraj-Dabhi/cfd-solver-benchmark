"""
OpenFOAM Log Parser
Extracts residuals, continuity errors, and execution time from OpenFOAM logs.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

class FoamLogParser:
    """Parses OpenFOAM standard output logs for analysis."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.time: List[float] = []
        # Store dict of lists for residuals, e.g., self.residuals['Ux'] = [0.1, 0.05, ...]
        self.residuals: Dict[str, List[float]] = {}
        self.continuity_errors: List[float] = []
        self.courant_numbers: List[float] = []
        self.execution_times: List[float] = []

    def parse(self) -> bool:
        """Parse the log file to extract residuals and timing."""
        if not self.log_path.exists():
            return False

        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Regular expressions for key OpenFOAM log patterns
        time_pattern = re.compile(r'^Time = ([0-9.eE+-]+)')
        solver_pattern = re.compile(r'^Solving for ([a-zA-Z0-9_.]+), Initial residual = ([0-9.eE+-]+), Final residual = ([0-9.eE+-]+), No Iterations ([0-9]+)')
        continuity_pattern = re.compile(r'^time step continuity errors : sum local = ([0-9.eE+-]+), global = ([0-9.eE+-]+), cumulative = ([0-9.eE+-]+)')
        courant_pattern = re.compile(r'^Courant Number mean: ([0-9.eE+-]+) max: ([0-9.eE+-]+)')
        if_courant_pattern = re.compile(r'^Interface Courant Number mean: ([0-9.eE+-]+) max: ([0-9.eE+-]+)')
        exec_time_pattern = re.compile(r'^ExecutionTime = ([0-9.eE+-]+) s')

        current_time = 0.0
        temp_residuals = {}

        for line in lines:
            line = line.strip()
            
            # Match Time
            m_time = time_pattern.match(line)
            if m_time:
                # Flush temp_residuals into main BEFORE advancing time
                for k, v in temp_residuals.items():
                    if k not in self.residuals:
                        # Back-fill with NaNs for missed early steps if needed
                        self.residuals[k] = [np.nan] * max(0, len(self.time) - 1)
                    self.residuals[k].append(v)
                temp_residuals = {}

                current_time = float(m_time.group(1))
                self.time.append(current_time)
                continue

            # Match solver residuals
            m_solver = solver_pattern.match(line)
            if m_solver:
                field = m_solver.group(1)
                init_res = float(m_solver.group(2))
                temp_residuals[field] = init_res
                continue

            # Match continuity
            m_cont = continuity_pattern.match(line)
            if m_cont:
                local_error = float(m_cont.group(1))
                self.continuity_errors.append(local_error)
                continue

            # Match Courant number
            m_cou = courant_pattern.match(line)
            if m_cou:
                max_co = float(m_cou.group(2))
                self.courant_numbers.append(max_co)
                continue

            # Match Execution Time
            m_exec = exec_time_pattern.match(line)
            if m_exec:
                exec_t = float(m_exec.group(1))
                self.execution_times.append(exec_t)

        # Flush final timestep residuals
        for k, v in temp_residuals.items():
            if k not in self.residuals:
                self.residuals[k] = [np.nan] * (len(self.time) - 1)
            self.residuals[k].append(v)

        return True

    def get_summary(self) -> Dict[str, Any]:
        """Return a statistical summary of the run."""
        if not self.time:
            return {"error": "Log not parsed or empty"}

        summary = {
            "total_timesteps": len(self.time),
            "final_time": self.time[-1],
        }

        if self.execution_times:
            summary["total_execution_time_s"] = self.execution_times[-1]

        if self.courant_numbers:
            summary["max_courant"] = max(self.courant_numbers)
            summary["mean_courant"] = np.mean(self.courant_numbers[len(self.courant_numbers)//2:])

        if self.continuity_errors:
            summary["final_continuity_error"] = self.continuity_errors[-1]

        # Final residuals
        final_res = {}
        for k, v_list in self.residuals.items():
            if v_list:
                final_res[k] = v_list[-1]
        summary["final_residuals"] = final_res

        return summary
