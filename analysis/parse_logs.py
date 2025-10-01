#!/usr/bin/env python3
import os, glob, argparse, csv, sys, re
from statistics import mean, pstdev

def parse_log_file(log_path):
    """
    parse a log file for lines like:
    'accuracies on testset after task X is: [acc1, acc2, ...] mean_acc'
    returns list of final accuracies per task, or None if not found
    """
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        return None
    
    # look for the pattern: "accuracies on testset after task X is: [...]"
    pattern = r'accuracies on testset after task \d+ is:\s*\[([\d\.,\s]+)\]\s*([\d\.]+)'
    
    task_accs = []
    for line in lines:
        match = re.search(pattern, line)
        if match:
            # extract the list of accuracies
            acc_list_str = match.group(1)
            mean_acc = float(match.group(2))
            # parse the list
            accs = [float(x.strip()) for x in acc_list_str.split(',')]
            task_accs.append((accs, mean_acc))
    
    if not task_accs:
        return None
    
    # return the final task's accuracies (last one)
    return task_accs[-1]

def discover_log_files(base_dir):
    """
    find all .out, .log, .err files under base_dir
    """
    patterns = ['**/*.out', '**/*.log', '**/*.err']
    log_files = []
    for pattern in patterns:
        log_files.extend(glob.glob(os.path.join(base_dir, pattern), recursive=True))
    return sorted(set(log_files))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    ap = argparse.ArgumentParser(description="parse CL log files for final accuracies")
    ap.add_argument("--base", default= '../logs',
                    help="base directory to search for log files (defaults to ../results)")
    ap.add_argument("--logs", nargs="*", help="explicit log file paths")
    ap.add_argument("--csv-name", default="table1_accuracies.csv", 
                    help="filename for CSV saved in analysis/")
    ap.add_argument("--verbose", action="store_true", help="print per-file details")
    args = ap.parse_args()
    
    # determine log files
    log_files = []
    if args.logs:
        log_files = args.logs
    else:
        base = args.base
        if base is None:
            base = os.path.normpath(os.path.join(script_dir, "..", "results"))
        log_files = discover_log_files(base)
        
        if not log_files:
            print(f"[!] no log files found under: {base}", file=sys.stderr)
            print("    provide explicit --logs or check --base directory", file=sys.stderr)
            sys.exit(1)
    
    rows = []
    
    for log_path in log_files:
        if not os.path.isfile(log_path):
            continue
            
        result = parse_log_file(log_path)
        if result is None:
            if args.verbose:
                print(f"[skip] no accuracy data in: {log_path}")
            continue
        
        accs, mean_acc = result
        
        # try to infer dataset from path and filename
        dataset = "unknown"
        log_name = os.path.basename(log_path).lower()
        log_dir = os.path.dirname(log_path).lower()
        
        # Check both filename and directory path for dataset patterns
        search_text = log_name + " " + log_dir
        
        # CIFAR variants (order matters - check more specific patterns first)
        if "cifar100_prv" in search_text or "cifar100-prv" in search_text:
            dataset = "split_cifar100_prv"
        elif "cifar100" in search_text:
            dataset = "split_cifar100"
        elif "cifar_prv_loss_diff" in search_text or "cifar-prv-loss-diff" in search_text:
            dataset = "split_cifar_prv_loss_diff"
        elif "cifar_prv_qvendi" in search_text or "cifar-prv-qvendi" in search_text:
            dataset = "split_cifar_prv_qvendi"
        elif "cifar_loss_diff" in search_text or "cifar-loss-diff" in search_text:
            dataset = "split_cifar_loss_diff"
        elif "cifar_qvendi" in search_text or "cifar-qvendi" in search_text:
            dataset = "split_cifar_qvendi"
        elif "cifar_prv" in search_text or "cifar-prv" in search_text:
            dataset = "split_cifar_prv"
        elif "cifar" in search_text:
            dataset = "split_cifar"
        # MNIST variants
        elif "mnist_prv_loss_diff" in search_text or "mnist-prv-loss-diff" in search_text:
            dataset = "split_mnist_prv_loss_diff"
        elif "mnist_prv_qvendi" in search_text or "mnist-prv-qvendi" in search_text:
            dataset = "split_mnist_prv_qvendi"
        elif "mnist_loss_diff" in search_text or "mnist-loss-diff" in search_text:
            dataset = "split_mnist_loss_diff"
        elif "mnist_qvendi" in search_text or "mnist-qvendi" in search_text:
            dataset = "split_mnist_qvendi"
        elif "mnist_prv" in search_text or "mnist-prv" in search_text:
            dataset = "split_mnist_prv"
        elif "mnist" in search_text:
            dataset = "split_mnist"
        # Permuted MNIST variants
        elif "perm_prv" in search_text or "perm-prv" in search_text:
            dataset = "perm_mnist_prv"
        elif "perm" in search_text:
            dataset = "perm_mnist"
        
        rows.append({
            "dataset": dataset,
            "log_file": os.path.basename(log_path),
            "final_mean_acc": mean_acc,
            "per_task_accs": accs,
            "n_tasks": len(accs),
            "path": log_path
        })
        
        if args.verbose:
            print(f"[{dataset}] {os.path.basename(log_path)}: {mean_acc:.2f}% (tasks: {accs})")
    
    # group by dataset
    dataset_groups = {}
    for r in rows:
        dataset_groups.setdefault(r["dataset"], []).append(r["final_mean_acc"])
    
    # print summary
    if dataset_groups:
        print("\nper-dataset summary:")
        for ds in sorted(dataset_groups.keys()):
            vals = dataset_groups[ds]
            m = mean(vals)
            s = pstdev(vals) if len(vals) > 1 else 0.0
            print(f"{ds:<25} {m:.2f} ± {s:.2f} over {len(vals)} run(s)")
    
    # write CSV
    if rows:
        csv_path = os.path.join(script_dir, args.csv_name)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "log_file", "final_mean_acc", "n_tasks", "per_task_accs", "log_path"])
            for r in rows:
                w.writerow([r["dataset"], r["log_file"], 
                           f"{r['final_mean_acc']:.2f}",
                           r["n_tasks"],
                           ",".join([f"{x:.2f}" for x in r["per_task_accs"]]),
                           r["path"]])
        print(f"\nwrote {csv_path}")
    else:
        print("[!] no accuracy results found in any log files", file=sys.stderr)

if __name__ == "__main__":
    main()
