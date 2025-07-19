#!/usr/bin/env python3
"""
Script to analyze iterative SOLiD training results.

This script compares lie detector performance and deception rates across iterations
to determine if representations have changed to make it harder to train probes.
"""

import argparse
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_evaluation_results(experiment_dir: str) -> Dict[int, Dict[str, Any]]:
    """
    Load evaluation results from each iteration.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Dictionary mapping iteration number to evaluation results
    """
    results = {}
    experiment_path = Path(experiment_dir)
    
    # Find all iteration directories
    iteration_dirs = [d for d in experiment_path.iterdir() if d.is_dir() and d.name.startswith("iteration_")]
    
    for iteration_dir in iteration_dirs:
        iteration_num = int(iteration_dir.name.split("_")[1])
        eval_dir = iteration_dir / "eval"
        
        if eval_dir.exists():
            # Look for evaluation results
            eval_files = list(eval_dir.glob("*.json"))
            if eval_files:
                # Load the first JSON file found
                with open(eval_files[0], 'r') as f:
                    results[iteration_num] = json.load(f)
                    logger.info(f"Loaded results for iteration {iteration_num}")
            else:
                logger.warning(f"No evaluation results found for iteration {iteration_num}")
        else:
            logger.warning(f"No eval directory found for iteration {iteration_num}")
    
    return results


def analyze_lie_detector_performance(results: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze lie detector performance across iterations.
    
    Args:
        results: Dictionary of evaluation results by iteration
        
    Returns:
        DataFrame with lie detector performance metrics
    """
    performance_data = []
    
    for iteration, result in results.items():
        # Extract lie detector metrics
        if 'lie_detector_metrics' in result:
            metrics = result['lie_detector_metrics']
            performance_data.append({
                'iteration': iteration,
                'auc': metrics.get('auc', np.nan),
                'tpr': metrics.get('true_positive_rate', np.nan),
                'fpr': metrics.get('false_positive_rate', np.nan),
                'accuracy': metrics.get('accuracy', np.nan),
            })
        else:
            logger.warning(f"No lie detector metrics found for iteration {iteration}")
    
    return pd.DataFrame(performance_data)


def analyze_deception_rates(results: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze deception rates across iterations.
    
    Args:
        results: Dictionary of evaluation results by iteration
        
    Returns:
        DataFrame with deception rate metrics
    """
    deception_data = []
    
    for iteration, result in results.items():
        # Extract deception rate metrics
        if 'deception_metrics' in result:
            metrics = result['deception_metrics']
            deception_data.append({
                'iteration': iteration,
                'deception_rate': metrics.get('deception_rate', np.nan),
                'detected_deception_rate': metrics.get('detected_deception_rate', np.nan),
                'undetected_deception_rate': metrics.get('undetected_deception_rate', np.nan),
                'honest_rate': metrics.get('honest_rate', np.nan),
            })
        else:
            logger.warning(f"No deception metrics found for iteration {iteration}")
    
    return pd.DataFrame(deception_data)


def create_comparison_plots(performance_df: pd.DataFrame, deception_df: pd.DataFrame, output_dir: str):
    """
    Create comparison plots for analysis.
    
    Args:
        performance_df: DataFrame with lie detector performance
        deception_df: DataFrame with deception rates
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Lie Detector AUC across iterations
    if not performance_df.empty:
        axes[0, 0].plot(performance_df['iteration'], performance_df['auc'], 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Lie Detector AUC')
        axes[0, 0].set_title('Lie Detector Performance (AUC)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: True Positive Rate vs False Positive Rate
        axes[0, 1].scatter(performance_df['fpr'], performance_df['tpr'], 
                          s=100, c=performance_df['iteration'], cmap='viridis')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('TPR vs FPR by Iteration')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add colorbar
        scatter = axes[0, 1].scatter(performance_df['fpr'], performance_df['tpr'], 
                                   s=100, c=performance_df['iteration'], cmap='viridis')
        plt.colorbar(scatter, ax=axes[0, 1], label='Iteration')
    
    # Plot 3: Deception rates across iterations
    if not deception_df.empty:
        x = deception_df['iteration']
        width = 0.35
        
        axes[1, 0].bar(x - width/2, deception_df['deception_rate'], width, 
                      label='Total Deception', alpha=0.8)
        axes[1, 0].bar(x + width/2, deception_df['undetected_deception_rate'], width,
                      label='Undetected Deception', alpha=0.8)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Rate')
        axes[1, 0].set_title('Deception Rates by Iteration')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Honest vs Deceptive responses
        axes[1, 1].bar(x - width/2, deception_df['honest_rate'], width,
                      label='Honest', alpha=0.8, color='green')
        axes[1, 1].bar(x + width/2, deception_df['deception_rate'], width,
                      label='Deceptive', alpha=0.8, color='red')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].set_title('Honest vs Deceptive Responses')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'iteration_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison plots saved to {output_path / 'iteration_comparison.png'}")


def generate_analysis_report(performance_df: pd.DataFrame, deception_df: pd.DataFrame, output_dir: str):
    """
    Generate a text report summarizing the analysis.
    
    Args:
        performance_df: DataFrame with lie detector performance
        deception_df: DataFrame with deception rates
        output_dir: Directory to save the report
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ITERATIVE SOLiD TRAINING ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Summary statistics
    if not performance_df.empty:
        report_lines.append("LIE DETECTOR PERFORMANCE ANALYSIS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Number of iterations analyzed: {len(performance_df)}")
        report_lines.append("")
        
        # AUC analysis
        if 'auc' in performance_df.columns:
            auc_values = performance_df['auc'].dropna()
            if len(auc_values) > 1:
                auc_change = auc_values.iloc[-1] - auc_values.iloc[0]
                report_lines.append(f"AUC Change: {auc_change:.4f}")
                report_lines.append(f"Initial AUC: {auc_values.iloc[0]:.4f}")
                report_lines.append(f"Final AUC: {auc_values.iloc[-1]:.4f}")
                
                if auc_change < -0.05:
                    report_lines.append("→ SIGNIFICANT DECREASE in lie detector performance")
                elif auc_change > 0.05:
                    report_lines.append("→ SIGNIFICANT INCREASE in lie detector performance")
                else:
                    report_lines.append("→ MINIMAL CHANGE in lie detector performance")
            report_lines.append("")
    
    if not deception_df.empty:
        report_lines.append("DECEPTION RATE ANALYSIS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Number of iterations analyzed: {len(deception_df)}")
        report_lines.append("")
        
        # Deception rate analysis
        if 'deception_rate' in deception_df.columns:
            deception_values = deception_df['deception_rate'].dropna()
            if len(deception_values) > 1:
                deception_change = deception_values.iloc[-1] - deception_values.iloc[0]
                report_lines.append(f"Deception Rate Change: {deception_change:.4f}")
                report_lines.append(f"Initial Deception Rate: {deception_values.iloc[0]:.4f}")
                report_lines.append(f"Final Deception Rate: {deception_values.iloc[-1]:.4f}")
                
                if deception_change > 0.1:
                    report_lines.append("→ SIGNIFICANT INCREASE in deception")
                elif deception_change < -0.1:
                    report_lines.append("→ SIGNIFICANT DECREASE in deception")
                else:
                    report_lines.append("→ MINIMAL CHANGE in deception rate")
            report_lines.append("")
    
    # Key findings
    report_lines.append("KEY FINDINGS:")
    report_lines.append("-" * 40)
    
    if not performance_df.empty and not deception_df.empty:
        # Check if representations have changed to make detection harder
        if 'auc' in performance_df.columns and 'deception_rate' in deception_df.columns:
            auc_values = performance_df['auc'].dropna()
            deception_values = deception_df['deception_rate'].dropna()
            
            if len(auc_values) > 1 and len(deception_values) > 1:
                auc_decrease = auc_values.iloc[-1] < auc_values.iloc[0]
                deception_increase = deception_values.iloc[-1] > deception_values.iloc[0]
                
                if auc_decrease and deception_increase:
                    report_lines.append("→ EVIDENCE OF REPRESENTATION CHANGE:")
                    report_lines.append("  - Lie detector performance decreased")
                    report_lines.append("  - Deception rate increased")
                    report_lines.append("  - This suggests the model learned to evade detection")
                elif not auc_decrease and deception_increase:
                    report_lines.append("→ PARTIAL EVIDENCE OF REPRESENTATION CHANGE:")
                    report_lines.append("  - Deception rate increased but detector performance maintained")
                    report_lines.append("  - Model may be learning more sophisticated deception")
                elif auc_decrease and not deception_increase:
                    report_lines.append("→ DETECTOR DEGRADATION:")
                    report_lines.append("  - Lie detector performance decreased")
                    report_lines.append("  - But deception rate did not increase")
                    report_lines.append("  - May indicate detector overfitting or data shift")
                else:
                    report_lines.append("→ NO CLEAR EVIDENCE OF REPRESENTATION CHANGE:")
                    report_lines.append("  - Both detector performance and deception rates stable")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    report_path = output_path / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Also print to console
    print('\n'.join(report_lines))
    
    logger.info(f"Analysis report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze iterative SOLiD training results")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to the experiment directory containing iteration results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis_output",
        help="Directory to save analysis results"
    )
    
    args = parser.parse_args()
    
    # Load results
    logger.info(f"Loading results from {args.experiment_dir}")
    results = load_evaluation_results(args.experiment_dir)
    
    if not results:
        logger.error("No evaluation results found!")
        return
    
    logger.info(f"Loaded results for {len(results)} iterations")
    
    # Analyze performance
    performance_df = analyze_lie_detector_performance(results)
    deception_df = analyze_deception_rates(results)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save dataframes
    if not performance_df.empty:
        performance_df.to_csv(output_path / 'lie_detector_performance.csv', index=False)
        logger.info("Saved lie detector performance data")
    
    if not deception_df.empty:
        deception_df.to_csv(output_path / 'deception_rates.csv', index=False)
        logger.info("Saved deception rate data")
    
    # Create plots
    create_comparison_plots(performance_df, deception_df, args.output_dir)
    
    # Generate report
    generate_analysis_report(performance_df, deception_df, args.output_dir)
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main() 