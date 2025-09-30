#!/usr/bin/env python3
import os, glob, pickle, argparse, csv
from statistics import mean, pstdev

ACC_KEYS = [
    "acc_per_task", "acc_list", "test_acc_list", "task_acc",
    "eval_accs", "accs"
]
AVG_KEYS = ["final_acc", "avg_acc"]

def extract_accs(path):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        return None

    if isinstance(obj, dict):
        for k in ACC_KEYS:
            if k in obj:
                v = obj[k]
                if isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float)):
                    return [float(x) for x in v]
                if isinstance(v, dict):
                    try:
                        keys = sorted([kk for kk in v.keys() if isinstance(kk, int)])
                        if keys:
                            return [float(v[i]) for i in keys]
                    except Exception:
                        pass
        for k in AVG_KEYS:
            if k in obj and isinstance(obj[k], (int, float)):
                return [float(obj[k])]
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], (int, float)):
        return [float(x) for x in obj]
    return None

def summarize_buffer_dir(buffer_dir, verbose=False):
    pkls = sorted(glob.glob(os.path.join(buffer_dir, "*.pkl")))
    per_file_avg, details = [], []
    for p in pkls:
        accs = extract_accs(p)
        if accs:
            avg_this_file = mean(accs)
            per_file_avg.append(avg_this_file)
            if verbose:
                details.append((os.path.basename(p), avg_this_file, len(accs)))
    if not per_file_avg:
        return None
    m = mean(per_file_avg)
    s = pstdev(per_file_avg) if len(per_file_avg) > 1 else 0.0
    return (m, s, len(per_file_avg), details)

def main():
    ap = argparse.ArgumentParser(description="Summarize CL runs into Table-1 style metrics.")
    ap.add_argument("roots", nargs="+",
                    help="Dataset directories to scan (e.g. ../results/split_cifar ...).")
    ap.add_argument("--verbose", action="store_true", help="Print per-file stats.")
    args = ap.parse_args()

    rows, dset_groups = [], {}

    for root in args.roots:
        if not os.path.isdir(root):
            continue
        testdirs = sorted([d for d in os.listdir(root) if d.startswith("test")])
        for t in testdirs:
            buffer_dir = os.path.join(root, t, "buffer")
            if not os.path.isdir(buffer_dir):
                continue
            summary = summarize_buffer_dir(buffer_dir, verbose=args.verbose)
            if summary is None:
                continue
            avg, std, nfiles, details = summary
            rows.append({"dataset": root, "test": t, "mean_acc": avg,
                         "std_acc": std, "n_files": nfiles, "path": buffer_dir})
            dset_groups.setdefault(root, []).append(avg)

            if args.verbose:
                print(f"[{root}/{t}]  mean={avg:.2f} ± {std:.2f} over {nfiles} pkl(s)")
                for fn, a, k in details:
                    print(f"   - {fn}: avg={a:.2f} (from {k} accs)")

    if rows:
        print("\nPer-test summary:")
        for r in rows:
            print(f"{r['dataset']}/{r['test']:<8}  mean={r['mean_acc']:.2f} ± {r['std_acc']:.2f}  (n={r['n_files']})")

    if dset_groups:
        print("\nPer-dataset aggregate (mean of test means):")
        for ds, vals in dset_groups.items():
            m, s = mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)
            print(f"{ds:<16} {m:.2f} ± {s:.2f} over {len(vals)} test(s)")

    if rows:
        # save CSV inside the analysis folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "table1_metrics.csv")

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset","test","mean_acc","std_acc","n_files","path"])
            for r in rows:
                w.writerow([r["dataset"], r["test"],
                            f"{r['mean_acc']:.6f}", f"{r['std_acc']:.6f}",
                            r["n_files"], r["path"]])
        print(f"\nWrote {csv_path}")

if __name__ == "__main__":
    main()
