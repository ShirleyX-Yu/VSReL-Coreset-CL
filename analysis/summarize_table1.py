#!/usr/bin/env python3
import os, glob, pickle, argparse, csv, sys
from statistics import mean, pstdev

# keys commonly used by CL repos for per-task or averaged accuracies
ACC_KEYS = ["acc_per_task", "acc_list", "test_acc_list", "task_acc", "eval_accs", "accs"]
AVG_KEYS = ["final_acc", "avg_acc"]

def extract_accs(path):
    """Return list of per-task accuracies (floats), or [avg] if only an average is present; else None."""
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        return None

    if isinstance(obj, dict):
        # explicit arrays
        for k in ACC_KEYS:
            if k in obj:
                v = obj[k]
                if isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float)):
                    return [float(x) for x in v]
                if isinstance(v, dict):
                    # sometimes keyed by int task ids
                    ikeys = [kk for kk in obj[k].keys() if isinstance(kk, int)]
                    if ikeys:
                        return [float(obj[k][i]) for i in sorted(ikeys)]
        # single-number averages
        for k in AVG_KEYS:
            if k in obj and isinstance(obj[k], (int, float)):
                return [float(obj[k])]
    # bare list of floats - check first element is actually a number
    if isinstance(obj, (list, tuple)) and obj:
        try:
            # try to convert first element to float to verify it's numeric
            if isinstance(obj[0], (int, float)):
                return [float(x) for x in obj]
        except (TypeError, ValueError):
            pass
    return None

def summarize_buffer_dir(buffer_dir, use_last=False, verbose=False):
    """Summarize one test's buffer dir: return (mean, std, n_files, details) or None if no metrics found."""
    pkls = sorted(glob.glob(os.path.join(buffer_dir, "*.pkl")))
    per_file_metric, details = [], []
    for p in pkls:
        accs = extract_accs(p)
        if accs:
            metric = accs[-1] if use_last else mean(accs)
            per_file_metric.append(metric)
            if verbose:
                details.append((os.path.basename(p), metric, len(accs)))
    if not per_file_metric:
        return None
    m = mean(per_file_metric)
    s = pstdev(per_file_metric) if len(per_file_metric) > 1 else 0.0
    return (m, s, len(per_file_metric), details)

def discover_roots(base_dir):
    """
    Auto-discover dataset roots under base_dir.
    A root is any immediate child of base_dir that contains at least one test*/buffer or seed*/buffer directory.
    """
    roots = []
    if not os.path.isdir(base_dir):
        return roots
    for name in sorted(os.listdir(base_dir)):
        candidate = os.path.join(base_dir, name)
        if not os.path.isdir(candidate):
            continue
        # look for test*/buffer or seed*/buffer one level down
        found = False
        for testdir in os.listdir(candidate):
            if testdir.startswith("test") or testdir.startswith("seed"):
                if os.path.isdir(os.path.join(candidate, testdir, "buffer")):
                    found = True
                    break
        if found:
            roots.append(candidate)
    return roots

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(description="Summarize CL runs into Table-1 style metrics.")
    ap.add_argument("roots", nargs="*", help="Explicit dataset roots (e.g., ../results/split_cifar ...).")
    ap.add_argument("--base", default=None,
                    help="Base directory containing dataset folders (defaults to ../results relative to this script).")
    ap.add_argument("--use-last", action="store_true",
                    help="Use last-task accuracy instead of mean over tasks.")
    ap.add_argument("--verbose", action="store_true", help="Print per-file stats.")
    ap.add_argument("--csv-name", default="table1_metrics.csv", help="Filename for CSV saved in analysis/.")
    args = ap.parse_args()

    # determine roots: explicit > auto-discovered under base > auto-discovered under ../results
    roots = []
    if args.roots:
        roots = args.roots
    else:
        base = args.base
        if base is None:
            base = os.path.normpath(os.path.join(script_dir, "..", "results"))
        roots = discover_roots(base)
        if not roots:
            print(f"[!] No dataset roots found. Tried base: {base}", file=sys.stderr)
            print("    Provide explicit roots, or set --base to the folder that contains your dataset subfolders.", file=sys.stderr)
            sys.exit(1)

    rows, dset_groups = [], {}

    for root in roots:
        if not os.path.isdir(root):
            print(f"[! ] Skipping non-directory root: {root}", file=sys.stderr)
            continue
        testdirs = sorted([d for d in os.listdir(root) if d.startswith("test") or d.startswith("seed")])
        any_found = False
        for t in testdirs:
            buffer_dir = os.path.join(root, t, "buffer")
            if not os.path.isdir(buffer_dir):
                continue
            summary = summarize_buffer_dir(buffer_dir, use_last=args.use_last, verbose=args.verbose)
            if summary is None:
                continue
            any_found = True
            avg, std, nfiles, details = summary
            rows.append({
                "dataset": root,
                "test": t,
                "metric": avg,
                "std": std,
                "n_files": nfiles,
                "path": buffer_dir
            })
            dset_groups.setdefault(root, []).append(avg)

            if args.verbose:
                print(f"[{root}/{t}]  {'last' if args.use_last else 'mean'}={avg:.2f} ± {std:.2f} over {nfiles} pkl(s)")
                for fn, val, k in details:
                    print(f"   - {fn}: {('last' if args.use_last else 'avg')}={val:.2f} (from {k} accs)")

        if not any_found and args.verbose:
            print(f"[i ] No metric-like pkls found under: {root}", file=sys.stderr)

    # pretty print
    if rows:
        print("\nPer-test summary:")
        label = "Last Acc" if args.use_last else "Mean Acc"
        for r in rows:
            print(f"{r['dataset']}/{r['test']:<8}  {label}={r['metric']:.2f} ± {r['std']:.2f}  (n={r['n_files']})")

    if dset_groups:
        print("\nPer-dataset aggregate (mean of test means):")
        for ds, vals in dset_groups.items():
            m, s = mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)
            print(f"{ds:<40} {m:.2f} ± {s:.2f} over {len(vals)} test(s)")

    # write CSV into the analysis folder alongside this script
    if rows:
        csv_path = os.path.join(script_dir, args.csv_name)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            # column names mirror what we print
            w.writerow(["dataset_root","test","metric","std","n_files","buffer_path","mode"])
            for r in rows:
                w.writerow([r["dataset"], r["test"],
                            f"{r['metric']:.6f}", f"{r['std']:.6f}",
                            r["n_files"], r["path"],
                            "last" if args.use_last else "mean"])
        print(f"\nWrote {csv_path}")

if __name__ == "__main__":
    main()
