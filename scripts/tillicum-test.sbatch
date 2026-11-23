#!/usr/bin/env bash
# Run the prebuilt regression suites on one Tillicum MIG instance.
#
# Build first, on a login node (no GPU time is billed for that):
#
#     source scripts/tillicum-env.sh
#     pixi run test -b -j 4          # serial tests  -> build/tests/
#     pixi run test-mpi -b -j 4      # MPI tests     -> build/mpi-tests/
#
# then submit from the repo root:
#
#     sbatch scripts/tillicum-test.sbatch            # both suites
#     sbatch scripts/tillicum-test.sbatch serial     # only tests/
#     sbatch scripts/tillicum-test.sbatch mpi        # only tests/mpi/
#
# A 1g.18gb MIG instance bills at 1/7 of a full H200 and the tests are
# small. The debug QoS caps a job at 1 GPU and 1 hour, and the site
# allows at most 2 CPUs per GPU (`sbatch --test-only` checks a request
# and prints its estimated cost). All MPI ranks share the one instance,
# as on a workstation, oversubscribing the 2 CPUs. If the GPU-aware
# halo path misbehaves on MIG, resubmit with WXM_GPU_AWARE_MPI=0 in the
# environment to stage halos through host memory instead.
#
#SBATCH --job-name=warpxm-tests
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=debug
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=build/slurm-%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/tillicum-env.sh

which="${1:-all}"
echo "job $SLURM_JOB_ID on $(hostname -s), $SLURM_CPUS_PER_TASK CPUs"
nvidia-smi -L

rc=0
if [[ "$which" == all || "$which" == serial ]]; then
    echo; echo "=== serial suite ==="
    pixi run --frozen test -r -v -t || rc=1
fi
if [[ "$which" == all || "$which" == mpi ]]; then
    echo; echo "=== MPI suite ==="
    pixi run --frozen test-mpi -r -v || rc=1
fi
exit $rc
