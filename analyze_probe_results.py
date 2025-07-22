#!/usr/bin/env python3
"""
Quick script to analyze probe detection results and print examples.
"""

import pandas as pd
import argparse
from pathlib import Path

def analyze_probe_results(csv_path):
    """Analyze probe detection results and print examples."""
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"\nTotal examples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check if probe detection columns exist
    if 'probe1_detected' not in df.columns or 'probe2_detected' not in df.columns:
        print("❌ Error: Probe detection columns not found!")
        print("Available columns:", list(df.columns))
        return
    
    # Count detection patterns
    both_detected = df[(df['probe1_detected'] == True) & (df['probe2_detected'] == True)]
    neither_detected = df[(df['probe1_detected'] == False) & (df['probe2_detected'] == False)]
    only_probe1 = df[(df['probe1_detected'] == True) & (df['probe2_detected'] == False)]
    only_probe2 = df[(df['probe1_detected'] == False) & (df['probe2_detected'] == True)]
    
    print(f"\n📊 Detection Summary:")
    print(f"  Both probes detected: {len(both_detected)} ({len(both_detected)/len(df)*100:.1f}%)")
    print(f"  Neither probe detected: {len(neither_detected)} ({len(neither_detected)/len(df)*100:.1f}%)")
    print(f"  Only Probe 1 detected: {len(only_probe1)} ({len(only_probe1)/len(df)*100:.1f}%)")
    print(f"  Only Probe 2 detected: {len(only_probe2)} ({len(only_probe2)/len(df)*100:.1f}%)")
    
    # Print examples caught by both probes
    print(f"\n🔍 EXAMPLES CAUGHT BY BOTH PROBES ({len(both_detected)} examples):")
    print("=" * 80)
    
    for i, (_, row) in enumerate(both_detected.head(5).iterrows()):
        print(f"\nExample {i+1}:")
        print(f"  Prompt: {row['prompt'][:200]}...")
        print(f"  Truthful Response: {row['truthful_response'][:200]}...")
        print(f"  Deceptive Response: {row['deceptive_response'][:200]}...")
        if 'probe1_score' in df.columns and 'probe2_score' in df.columns:
            print(f"  Probe 1 Score: {row['probe1_score']:.4f}")
            print(f"  Probe 2 Score: {row['probe2_score']:.4f}")
        print("-" * 40)
    
    if len(both_detected) > 5:
        print(f"  ... and {len(both_detected) - 5} more examples")
    
    # Print examples caught by neither probe
    print(f"\n🚫 EXAMPLES CAUGHT BY NEITHER PROBE ({len(neither_detected)} examples):")
    print("=" * 80)
    
    for i, (_, row) in enumerate(neither_detected.head(5).iterrows()):
        print(f"\nExample {i+1}:")
        print(f"  Prompt: {row['prompt'][:200]}...")
        print(f"  Truthful Response: {row['truthful_response'][:200]}...")
        print(f"  Deceptive Response: {row['deceptive_response'][:200]}...")
        if 'probe1_score' in df.columns and 'probe2_score' in df.columns:
            print(f"  Probe 1 Score: {row['probe1_score']:.4f}")
            print(f"  Probe 2 Score: {row['probe2_score']:.4f}")
        print("-" * 40)
    
    if len(neither_detected) > 5:
        print(f"  ... and {len(neither_detected) - 5} more examples")

def main():
    parser = argparse.ArgumentParser(description="Analyze probe detection results")
    parser.add_argument("csv_path", type=str, help="Path to CSV with probe predictions")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"❌ Error: File not found: {csv_path}")
        return
    
    analyze_probe_results(csv_path)

if __name__ == "__main__":
    main() 