#!/usr/bin/env python3
"""
Test script to verify iterative data splitting works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'solid_deception'))

from data_generation.munge_data import create_iterative_splits
from datasets import load_dataset
import numpy as np

def test_iterative_splitting():
    """Test the iterative splitting function."""
    print("Testing iterative data splitting...")
    
    # Load a small subset of the dataset for testing
    try:
        dataset = load_dataset('AlignmentResearch/DolusChat')['train']
        # Use only first 1000 examples for testing
        dataset = dataset.select(range(min(1000, len(dataset))))
        print(f"Loaded {len(dataset)} examples for testing")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Test parameters
    h1_frac = 0.5
    test_frac = 0.05
    train_lr_frac = 0.05
    seed = 42
    
    print(f"Testing with h1_frac={h1_frac}, test_frac={test_frac}, train_lr_frac={train_lr_frac}")
    
    # Test iteration 1
    print("\n--- Testing Iteration 1 ---")
    h1_train, h1_train_lr, h1_test, h2 = create_iterative_splits(
        dataset, h1_frac, test_frac, train_lr_frac, iteration=1, seed=seed
    )
    
    print(f"h1_train: {len(h1_train)} examples")
    print(f"h1_train_lr: {len(h1_train_lr)} examples")
    print(f"h1_test: {len(h1_test)} examples")
    print(f"h2: {len(h2)} examples")
    
    # Verify splits
    total_h1 = len(h1_train) + len(h1_train_lr) + len(h1_test)
    expected_h1 = int(len(dataset) * h1_frac)
    print(f"Total h1 examples: {total_h1} (expected: {expected_h1})")
    
    if total_h1 != expected_h1:
        print("❌ h1 split size mismatch!")
        return False
    
    # Test iteration 2
    print("\n--- Testing Iteration 2 ---")
    h2_train, h2_train_lr, h2_test, h1 = create_iterative_splits(
        dataset, h1_frac, test_frac, train_lr_frac, iteration=2, seed=seed
    )
    
    print(f"h2_train: {len(h2_train)} examples")
    print(f"h2_train_lr: {len(h2_train_lr)} examples")
    print(f"h2_test: {len(h2_test)} examples")
    print(f"h1: {len(h1)} examples")
    
    # Verify splits
    total_h2 = len(h2_train) + len(h2_train_lr) + len(h2_test)
    expected_h2 = len(dataset) - expected_h1
    print(f"Total h2 examples: {total_h2} (expected: {expected_h2})")
    
    if total_h2 != expected_h2:
        print("❌ h2 split size mismatch!")
        return False
    
    # Verify no overlap between h1 and h2
    h1_ids = set(h1['id'] for h1 in h1)
    h2_ids = set(h2['id'] for h2 in h2)
    overlap = h1_ids.intersection(h2_ids)
    
    if overlap:
        print(f"❌ Found {len(overlap)} overlapping examples between h1 and h2!")
        return False
    else:
        print("✅ No overlap between h1 and h2 splits")
    
    # Verify all examples are accounted for
    total_accounted = total_h1 + total_h2
    if total_accounted != len(dataset):
        print(f"❌ Total accounted examples ({total_accounted}) != dataset size ({len(dataset)})")
        return False
    else:
        print("✅ All examples accounted for")
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_iterative_splitting()
    sys.exit(0 if success else 1) 