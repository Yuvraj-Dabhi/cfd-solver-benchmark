#!/bin/bash
#==============================================================================
# CFD Solver Benchmark – SLURM HPC Submission Script
#==============================================================================
# Usage:
#   sbatch hpc_submit.sh <case_dir> <solver> <model> <n_procs>
#
# Examples:
#   sbatch hpc_submit.sh ./cases/bfs_SST_fine simpleFoam SST 16
#   sbatch hpc_submit.sh ./cases/hump_SA_medium SU2_CFD SA 8
#
# Environment variables (override defaults):
#   CFD_PARTITION, CFD_WALLTIME, CFD_ACCOUNT, CFD_QOS
#==============================================================================

#SBATCH --job-name=cfd-bench
#SBATCH --partition=${CFD_PARTITION:-standard}
#SBATCH --account=${CFD_ACCOUNT:-default}
#SBATCH --qos=${CFD_QOS:-normal}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${4:-4}
#SBATCH --time=${CFD_WALLTIME:-04:00:00}
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm-%j-%x.out
#SBATCH --error=slurm-%j-%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@university.edu

set -euo pipefail

# =============================================================================
# Arguments
# =============================================================================
CASE_DIR="${1:?Usage: sbatch hpc_submit.sh <case_dir> <solver> <model> <n_procs>}"
SOLVER="${2:-simpleFoam}"
MODEL="${3:-SST}"
NPROCS="${4:-$SLURM_NTASKS}"

echo "============================================"
echo "  CFD Solver Benchmark – HPC Job"
echo "============================================"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Node:       $SLURM_NODELIST"
echo "  Tasks:      $NPROCS"
echo "  Case:       $CASE_DIR"
echo "  Solver:     $SOLVER"
echo "  Model:      $MODEL"
echo "  Start time: $(date)"
echo "============================================"

# =============================================================================
# Environment Setup
# =============================================================================
# Load modules (adjust for your HPC)
module purge 2>/dev/null || true

if [[ "$SOLVER" == "SU2_CFD" ]]; then
    module load su2/8.0 2>/dev/null || true
    module load openmpi/4.1 2>/dev/null || true
else
    # OpenFOAM
    module load openfoam/2312 2>/dev/null || true
    source ${FOAM_INST_DIR:-/opt/openfoam}/etc/bashrc 2>/dev/null || true
fi

module load python/3.10 2>/dev/null || true

# Verify solver is available
if ! command -v "$SOLVER" &>/dev/null; then
    echo "ERROR: $SOLVER not found in PATH"
    exit 1
fi

cd "$CASE_DIR"

# =============================================================================
# Pre-processing
# =============================================================================
echo ""
echo "--- Pre-processing ---"

if [[ "$SOLVER" != "SU2_CFD" ]]; then
    # OpenFOAM: decompose for parallel
    if [[ $NPROCS -gt 1 ]]; then
        # Write decomposeParDict if not present
        if [[ ! -f system/decomposeParDict ]]; then
            cat > system/decomposeParDict <<EOF
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}
numberOfSubdomains  $NPROCS;
method              scotch;
EOF
        fi
        echo "Decomposing mesh for $NPROCS processors..."
        decomposePar -force > log.decomposePar 2>&1
    fi
fi

# =============================================================================
# Run Solver
# =============================================================================
echo ""
echo "--- Running $SOLVER ($NPROCS procs) ---"
START_TIME=$(date +%s)

if [[ "$SOLVER" == "SU2_CFD" ]]; then
    if [[ $NPROCS -gt 1 ]]; then
        mpirun -np "$NPROCS" SU2_CFD config.cfg > log.SU2_CFD 2>&1
    else
        SU2_CFD config.cfg > log.SU2_CFD 2>&1
    fi
else
    if [[ $NPROCS -gt 1 ]]; then
        mpirun -np "$NPROCS" "$SOLVER" -parallel > "log.$SOLVER" 2>&1
    else
        $SOLVER > "log.$SOLVER" 2>&1
    fi
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Solver completed in ${ELAPSED}s"

# =============================================================================
# Post-processing
# =============================================================================
echo ""
echo "--- Post-processing ---"

if [[ "$SOLVER" != "SU2_CFD" ]]; then
    # OpenFOAM: reconstruct + extract
    if [[ $NPROCS -gt 1 ]]; then
        echo "Reconstructing..."
        reconstructPar -latestTime > log.reconstructPar 2>&1
    fi

    echo "Extracting wall data..."
    postProcess -func wallShearStress -latestTime > log.wallShearStress 2>&1 || true
    postProcess -func yPlus -latestTime > log.yPlus 2>&1 || true

    # Check convergence
    FINAL_P_RES=$(grep "Solving for p" "log.$SOLVER" | tail -1 | \
        grep -oP 'Initial residual = \K[0-9.eE+-]+' || echo "N/A")
    echo "Final p residual: $FINAL_P_RES"
fi

# =============================================================================
# Cleanup
# =============================================================================
if [[ $NPROCS -gt 1 && "$SOLVER" != "SU2_CFD" ]]; then
    echo "Cleaning up processor directories..."
    rm -rf processor* 2>/dev/null || true
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================"
echo "  Job Complete"
echo "============================================"
echo "  End time:   $(date)"
echo "  Wall time:  ${ELAPSED}s"
echo "  Case:       $CASE_DIR"
echo "  Log file:   log.$SOLVER"
echo "============================================"

# Create a brief JSON summary
cat > job_summary.json <<EOF
{
    "job_id": "$SLURM_JOB_ID",
    "case_dir": "$CASE_DIR",
    "solver": "$SOLVER",
    "model": "$MODEL",
    "n_procs": $NPROCS,
    "wall_time_s": $ELAPSED,
    "node": "$SLURM_NODELIST",
    "timestamp": "$(date -Iseconds)"
}
EOF

echo "Summary written to job_summary.json"
