# Iterative SOLiD Training

This directory contains the implementation of iterative SOLiD (Scalable Oversight via Lie Detector) training, which allows you to investigate whether model representations change to make it harder to train lie detection probes across training iterations.

## Overview

The iterative SOLiD procedure implements the following workflow:

1. **Split dataset into h1 and h2**: The dataset is split into two halves
2. **Iteration 1**: Run SOLiD procedure with h1, resulting in policy p1
3. **Iteration 2**: Use p1 as the surrogate for the initial policy (instead of the original Llama model)
4. **Analysis**: Compare lie detector performance and deception rates across iterations

This allows us to answer the question: **Have the representations changed such that it is harder to train a probe to distinguish between truth and false answers?**

## Files

- `run_iterative.sh`: Main script for running iterative SOLiD training
- `analyze_iterations.py`: Script to analyze results across iterations
- `solid_deception/data_generation/munge_data.py`: Modified to support h1/h2 splitting

## Usage

### Running Iterative Training

```bash
# Run the iterative training procedure
./run_iterative.sh
```

The script will:
1. Create separate directories for each iteration
2. Split the data appropriately (h1 for iteration 1, h2 for iteration 2)
3. Use the trained policy from iteration 1 as the base model for iteration 2
4. Run the full SOLiD pipeline for each iteration
5. Save all results in organized directories

### Configuration

You can modify the following parameters in `run_iterative.sh`:

```bash
# Number of iterations (default: 2)
export NUM_ITERATIONS=2

# Fraction of data to use for h1 (default: 0.5)
export H1_FRAC=0.5

# Base model to start with (default: Llama-3.2-1B-Instruct)
export BASE_POLICY_PATH="meta-llama/Llama-3.2-1B-Instruct"

# Other training parameters (same as original run.sh)
export LIE_TPR=0.9
export DO_DPO=true
export USE_GRPO=false
# ... etc
```

### Analyzing Results

After training completes, analyze the results:

```bash
# Analyze results from a specific experiment
python analyze_iterations.py \
    --experiment_dir outputs/YYYYMMDD_HHMMSS \
    --output_dir analysis_output
```

This will generate:
- `lie_detector_performance.csv`: Performance metrics across iterations
- `deception_rates.csv`: Deception rates across iterations
- `iteration_comparison.png`: Visualization plots
- `analysis_report.txt`: Summary report with key findings

## Directory Structure

After running, you'll have:

```
outputs/
└── YYYYMMDD_HHMMSS/
    ├── iterative_stdout_err.log
    ├── iteration_1/
    │   ├── stdout_err.log
    │   ├── munged_data.csv
    │   ├── detected.csv
    │   ├── lr.pkl
    │   ├── rewarded/
    │   ├── rewarded_csv.csv
    │   ├── rm/
    │   ├── sft/
    │   ├── policy/
    │   └── eval/
    └── iteration_2/
        ├── stdout_err.log
        ├── munged_data.csv
        ├── detected.csv
        ├── lr.pkl
        ├── rewarded/
        ├── rewarded_csv.csv
        ├── rm/
        ├── sft/
        ├── policy/
        └── eval/
```

## Key Questions Answered

The analysis will help answer:

1. **Representation Change**: Did the model's internal representations change to make lie detection harder?
2. **Detection Evasion**: Did the model learn to evade the lie detector while maintaining deception?
3. **Probe Robustness**: How robust are lie detection probes across different model states?

## Expected Outcomes

Based on the paper's findings, we might expect:

- **If representations change**: Lie detector AUC decreases while deception rates increase
- **If no change**: Both metrics remain relatively stable
- **If detector degradation**: AUC decreases but deception rates don't increase

## Troubleshooting

### Common Issues

1. **Memory Issues**: If you run out of memory, try:
   - Reducing `BASE_PDTBS`
   - Using a smaller model
   - Setting `DEBUG_TRAINING=true` for smaller batches

2. **Training Failures**: Check the log files in each iteration directory for specific errors

3. **Analysis Issues**: Make sure the evaluation completed successfully before running analysis

### Logs

- Main log: `outputs/YYYYMMDD_HHMMSS/iterative_stdout_err.log`
- Per-iteration logs: `outputs/YYYYMMDD_HHMMSS/iteration_N/stdout_err.log`

## Research Implications

This iterative approach allows us to:

1. **Test Goodhart's Law**: See if models learn to game the lie detector
2. **Assess Scalability**: Understand if SOLiD remains effective across iterations
3. **Guide Policy**: Determine best practices for detector-in-the-loop training

The results will help inform whether SOLiD is a viable long-term approach for scalable oversight or if it inevitably leads to detection evasion. 