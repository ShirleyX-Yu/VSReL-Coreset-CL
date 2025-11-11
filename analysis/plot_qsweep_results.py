#!/usr/bin/env python3
"""
Plot Q-sweep results comparing Q-Vendi with different q values against loss_diff baseline.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "table1_accuracies.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run parse_logs.py first.", file=sys.stderr)
        sys.exit(1)
    
    # Read results
    df = pd.read_csv(csv_path)
    
    # Extract q values from dataset names
    df['base_dataset'] = df['dataset'].str.replace(r'_q[\d\.]+|_qinf', '', regex=True)
    df['q_value'] = df['dataset'].str.extract(r'_q([\d\.]+|inf)')[0]
    df['method'] = df['dataset'].apply(lambda x: 'qvendi' if 'qvendi' in x else 'loss_diff')
    
    # Convert q to numeric (inf -> large number for plotting)
    df['q_numeric'] = df['q_value'].replace('inf', '100').astype(float)
    
    # Get unique datasets
    datasets = df['base_dataset'].unique()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    dataset_names = {
        'split_mnist': 'Split MNIST',
        'perm_mnist': 'Permuted MNIST',
        'split_cifar': 'Split CIFAR-10',
        'split_cifar100': 'Split CIFAR-100'
    }
    
    for idx, dataset in enumerate(['split_mnist', 'perm_mnist', 'split_cifar', 'split_cifar100']):
        ax = axes[idx]
        
        # Filter data for this dataset
        dataset_data = df[df['base_dataset'] == dataset].copy()
        
        if len(dataset_data) == 0:
            ax.text(0.5, 0.5, f'No data for {dataset}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(dataset_names.get(dataset, dataset))
            continue
        
        # Plot Q-Vendi with different q values
        qvendi_data = dataset_data[dataset_data['method'] == 'qvendi'].sort_values('q_numeric')
        if len(qvendi_data) > 0:
            ax.plot(qvendi_data['q_numeric'], qvendi_data['final_mean_acc'], 
                   'o-', label='Q-Vendi', linewidth=2, markersize=8)
        
        # Plot loss_diff baseline as horizontal line
        loss_diff_data = dataset_data[dataset_data['method'] == 'loss_diff']
        if len(loss_diff_data) > 0:
            baseline_acc = loss_diff_data['final_mean_acc'].mean()
            ax.axhline(y=baseline_acc, color='red', linestyle='--', 
                      linewidth=2, label='Loss-Diff Baseline')
        
        ax.set_xlabel('q value', fontsize=11)
        ax.set_ylabel('Final Mean Accuracy (%)', fontsize=11)
        ax.set_title(dataset_names.get(dataset, dataset), fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks([0.1, 0.5, 1.0, 2.0, 10.0, 100])
        ax.set_xticklabels(['0.1', '0.5', '1.0', '2.0', '10.0', 'inf'])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(script_dir, 'qsweep_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    # Also create a summary table
    print("\n" + "="*80)
    print("Q-SWEEP RESULTS SUMMARY")
    print("="*80)
    
    for dataset in ['split_mnist', 'perm_mnist', 'split_cifar', 'split_cifar100']:
        dataset_data = df[df['base_dataset'] == dataset].copy()
        if len(dataset_data) == 0:
            continue
            
        print(f"\n{dataset_names.get(dataset, dataset)}:")
        print("-" * 60)
        
        # Loss-diff baseline
        loss_diff_data = dataset_data[dataset_data['method'] == 'loss_diff']
        if len(loss_diff_data) > 0:
            baseline_acc = loss_diff_data['final_mean_acc'].mean()
            print(f"  Loss-Diff Baseline: {baseline_acc:.2f}%")
        
        # Q-Vendi results
        qvendi_data = dataset_data[dataset_data['method'] == 'qvendi'].sort_values('q_numeric')
        if len(qvendi_data) > 0:
            print(f"  Q-Vendi results:")
            for _, row in qvendi_data.iterrows():
                q_val = row['q_value']
                acc = row['final_mean_acc']
                diff = acc - baseline_acc if len(loss_diff_data) > 0 else 0
                print(f"    q={q_val:>4s}: {acc:.2f}% (Δ{diff:+.2f}%)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
