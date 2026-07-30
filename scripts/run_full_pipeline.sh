#!/bin/bash
#SBATCH -A <your_allocation>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --job-name=hpc_tool_weekly
#SBATCH --output=logs/weekly_%j.log
#SBATCH --error=logs/full_pipeline_%j.err

# ═══════════════════════════════════════════════════════════════════════════
# Cross-Layer HPC Tool — Full Pipeline Rerun
# 
# This script runs the ENTIRE pipeline from scratch:
#   stage00_validate     → pre-flight data checks
#   00a             → GPU parquet conversion (skip if already done)
#   01              → GPU telemetry parsing
#   02              → Darshan log parsing (LONGEST STEP — hours)
#   stage00_validate     → post-extraction validation
#   03              → Build combined metrics
#   stage00_validate     → post-join validation
#   stage00_consistency  → cross-source consistency
#   04              → User fingerprints + phase patterns
#   05              → Repeatability analysis
#   06              → HTML results report
#   07              → Paper statistics
#   08              → User forensics (tar extraction — ~30min)
#   09              → Security audit
#   10              → ML predictive analysis
#   11              → Job report generator
#
# IMPORTANT: Replace darshan_utils.py BEFORE running this script:
#   cp darshan_utils_v2.py utils/darshan_utils.py
#
# Estimated runtime: 12-18 hours (dominated by step 02)
# ═══════════════════════════════════════════════════════════════════════════

# set -euo pipefail  # exit on error, undefined vars, pipe failures

# ── Setup ────────────────────────────────────────────────────────────────
PROJECT_DIR="/pscratch/sd/h/hjajula/Darshan/cross_layer_hpc_tool"
CONFIG="${PROJECT_DIR}/config/config.json"
LOG_DIR="${PROJECT_DIR}/logs"
RESULTS_DIR="${PROJECT_DIR}/results"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

# activate environment
module load python

echo "═══════════════════════════════════════════════════════════════"
echo "  Cross-Layer HPC Tool — Full Pipeline"
echo "  Started: $(date)"
echo "  Node: $(hostname)"
echo "  Config: ${CONFIG}"
echo "═══════════════════════════════════════════════════════════════"

# ── Helper function ──────────────────────────────────────────────────────
run_step() {
    local step_name="$1"
    local command="$2"
    local start_time=$(date +%s)
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP: ${step_name}"
    echo "  Time: $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    eval "${command}" || true
    local exit_code=${PIPESTATUS[0]:-$?}
    
    local end_time=$(date +%s)
    local duration=$(( end_time - start_time ))
    local minutes=$(( duration / 60 ))
    local seconds=$(( duration % 60 ))
    
    if [ $exit_code -eq 0 ]; then
        echo "  ✅ ${step_name} completed in ${minutes}m ${seconds}s"
    else
        echo "  ❌ ${step_name} FAILED (exit code: ${exit_code}) after ${minutes}m ${seconds}s"
        # don't exit on validation warnings (exit code 1)
        if [ $exit_code -gt 1 ]; then
            echo "  CRITICAL FAILURE — stopping pipeline"
            exit $exit_code
        fi
    fi
    
    echo "${step_name},${duration},${exit_code}" >> "${LOG_DIR}/pipeline_timing.csv"
}

# initialize timing log
echo "step,duration_seconds,exit_code" > "${LOG_DIR}/pipeline_timing.csv"

# ══════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT: Verify darshan_utils.py is updated
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "Pre-flight checks..."

# verify the new darshan_utils has the new fields
if grep -q "executable" utils/darshan_utils.py && \
   grep -q "posix_opens" utils/darshan_utils.py && \
   grep -q "unique_files" utils/darshan_utils.py; then
    echo "  ✅ darshan_utils.py has new extraction fields"
else
    echo "  ❌ darshan_utils.py is outdated — copy the updated version first:"
    echo "     cp darshan_utils_v2.py utils/darshan_utils.py"
    exit 1
fi

# verify config exists
if [ ! -f "${CONFIG}" ]; then
    echo "  ❌ Config not found: ${CONFIG}"
    exit 1
fi
echo "  ✅ Config found"

# verify darshan-parser is available
if ! command -v darshan-parser &> /dev/null; then
    echo "  ⚠️  darshan-parser not in PATH — loading module"
    module load darshan-runtime 2>/dev/null || module load darshan 2>/dev/null || true
    if ! command -v darshan-parser &> /dev/null; then
        echo "  ❌ darshan-parser still not available"
        exit 1
    fi
fi
echo "  ✅ darshan-parser available: $(which darshan-parser)"

# ══════════════════════════════════════════════════════════════════════════
# STAGE 0: Pre-flight validation of raw data
# ══════════════════════════════════════════════════════════════════════════

run_step "00_validate_preflight" \
    "python -m pipeline.stage00_validate --config ${CONFIG} --stage all 2>&1 | tee ${LOG_DIR}/00_validate_preflight.log"

# ══════════════════════════════════════════════════════════════════════════
# STAGE 0a: GPU Parquet conversion (skip if already done)
# ══════════════════════════════════════════════════════════════════════════

# Check if parquet data already exists
PARQUET_DIR=$(python -c "import json; print(json.load(open('${CONFIG}'))['parquet_out_dir'])")
PARQUET_COUNT=$(find "${PARQUET_DIR}" -name "*.parquet" 2>/dev/null | wc -l)

if [ "${PARQUET_COUNT}" -gt 100 ]; then
    echo ""
    echo "  ℹ️  Skipping GPU parquet conversion — ${PARQUET_COUNT} files already exist"
else
    run_step "stage00_convert_gpu_to_parquet" \
        "python -m pipeline.stage00_convert_gpu_to_parquet --config ${CONFIG} 2>&1 | tee ${LOG_DIR}/00_convert.log"
fi

# ══════════════════════════════════════════════════════════════════════════
# STAGE 1: GPU telemetry parsing
# ══════════════════════════════════════════════════════════════════════════

# Delete old output to force fresh parse
GPU_OUT=$(python -c "import json; print(json.load(open('${CONFIG}'))['gpu_parsed_out'])" 2>/dev/null || echo "")
if [ -n "${GPU_OUT}" ] && [ -f "${GPU_OUT}" ]; then
    echo "  Removing old GPU output: ${GPU_OUT}"
    rm -f "${GPU_OUT}"
fi

run_step "stage01_parse_gpu" \
    "python -m pipeline.stage01_parse_gpu --config ${CONFIG} 2>&1 | tee ${LOG_DIR}/stage01_parse_gpu.log"

run_step "00_validate_gpu" \
    "python -m pipeline.stage00_validate --config ${CONFIG} --stage gpu 2>&1 | tee ${LOG_DIR}/00_validate_gpu.log"

# ══════════════════════════════════════════════════════════════════════════
# STAGE 2: Darshan log parsing (LONGEST STEP)
# ══════════════════════════════════════════════════════════════════════════

# Delete old darshan output to force fresh parse with new fields
DARSHAN_OUT=$(python -c "import json; print(json.load(open('${CONFIG}'))['darshan_parsed_out'])" 2>/dev/null || echo "")
if [ -n "${DARSHAN_OUT}" ] && [ -f "${DARSHAN_OUT}" ]; then
    echo ""
    echo "  ⚠️  Removing old Darshan output to re-extract with new fields: ${DARSHAN_OUT}"
    echo "  ⚠️  This will take several hours to re-parse 843K+ files"
    echo ""
    # backup old file just in case
    cp "${DARSHAN_OUT}" "${DARSHAN_OUT}.bak.$(date +%Y%m%d_%H%M%S)"
    rm -f "${DARSHAN_OUT}"
fi

run_step "stage02_parse_darshan" \
    "python -m pipeline.stage02_parse_darshan --config ${CONFIG} 2>&1 | tee ${LOG_DIR}/stage02_parse_darshan.log"

run_step "00_validate_darshan" \
    "python -m pipeline.stage00_validate --config ${CONFIG} --stage darshan 2>&1 | tee ${LOG_DIR}/00_validate_darshan.log"

# ══════════════════════════════════════════════════════════════════════════
# STAGE 3: Build combined metrics
# ══════════════════════════════════════════════════════════════════════════

run_step "stage03_build_combined" \
    "python -m pipeline.stage03_build_combined --config ${CONFIG} 2>&1 | tee ${LOG_DIR}/03_combined.log"

run_step "00_validate_combined" \
    "python -m pipeline.stage00_validate --config ${CONFIG} --stage combined 2>&1 | tee ${LOG_DIR}/00_validate_combined.log"

run_step "stage00_consistency" \
    "python -m pipeline.stage00_consistency --config ${CONFIG} 2>&1 | tee ${LOG_DIR}/stage00_consistency.log"

# ══════════════════════════════════════════════════════════════════════════
# STAGE 4-6: Analysis pipeline
# ══════════════════════════════════════════════════════════════════════════

run_step "stage04_fingerprint" \
    "python -m pipeline.stage04_fingerprint --config ${CONFIG} 2>&1 | tee ${LOG_DIR}/stage04_fingerprint.log"

run_step "stage05_repeatability" \
    "python -m pipeline.stage05_repeatability --config ${CONFIG} 2>&1 | tee ${LOG_DIR}/stage05_repeatability.log"

run_step "stage06_results" \
    "python -m pipeline.stage06_results --config ${CONFIG} 2>&1 | tee ${LOG_DIR}/stage06_results.log"

# ══════════════════════════════════════════════════════════════════════════
# STAGE 7-11: Paper-specific analysis (not part of git)
# ══════════════════════════════════════════════════════════════════════════

run_step "stage07_paper_stats" \
    "python -m pipeline.stage07_paper_stats --config ${CONFIG} > ${RESULTS_DIR}/paper_stats.txt"

run_step "stage08_user_forensics" \
    "python -m pipeline.stage08_user_forensics --config ${CONFIG} > ${RESULTS_DIR}/user_forensics.txt"

run_step "stage09_security_audit" \
    "python -m pipeline.stage09_security_audit --config ${CONFIG} > ${RESULTS_DIR}/security_audit.txt"

run_step "stage10_predictive" \
    "python -m pipeline.stage10_predictive --config ${CONFIG} > ${RESULTS_DIR}/predictive.txt"

run_step "stage11_job_reports" \
    "python -m pipeline.stage11_job_reports --config ${CONFIG} > ${RESULTS_DIR}/job_reports.txt"

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "  Finished: $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# print timing summary
echo "Step Timing Summary:"
echo "────────────────────────────────────────────────────────────"
while IFS=',' read -r step duration exit_code; do
    if [ "$step" = "step" ]; then continue; fi
    minutes=$(( duration / 60 ))
    seconds=$(( duration % 60 ))
    status="✅"
    if [ "$exit_code" != "0" ]; then status="⚠️"; fi
    printf "  %s %-30s %4dm %2ds\n" "$status" "$step" "$minutes" "$seconds"
done < "${LOG_DIR}/pipeline_timing.csv"

total_seconds=$(awk -F',' 'NR>1 {sum+=$2} END {print sum}' "${LOG_DIR}/pipeline_timing.csv")
total_hours=$(echo "scale=1; $total_seconds / 3600" | bc)
echo "────────────────────────────────────────────────────────────"
echo "  Total: ${total_hours} hours"
echo ""

# list outputs
echo "Output files:"
echo "────────────────────────────────────────────────────────────"
echo "  Core pipeline:"
ls -lh data/combined_metrics.csv 2>/dev/null || echo "    combined_metrics.csv NOT FOUND"
ls -lh data/user_fingerprints.csv 2>/dev/null || true
ls -lh data/repeatability_scores.csv 2>/dev/null || true
ls -lh data/results_report.html 2>/dev/null || true
echo ""
echo "  Paper results:"
ls -lh ${RESULTS_DIR}/*.txt 2>/dev/null || echo "    No results files"
echo ""
echo "  Validation logs:"
ls -lh ${LOG_DIR}/00_*.log 2>/dev/null || echo "    No validation logs"
echo ""
echo "  Data artifacts:"
ls -lh data/ghost_anomaly_scores.csv 2>/dev/null || true
ls -lh data/user_forensics.csv 2>/dev/null || true
ls -lh data/ml_features.csv 2>/dev/null || true
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  All results in: ${RESULTS_DIR}/"
echo "  All logs in:    ${LOG_DIR}/"
echo "  HTML report:    data/results_report.html"
echo "═══════════════════════════════════════════════════════════════"
