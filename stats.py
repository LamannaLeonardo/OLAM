import json
import os

import pandas as pd

if __name__ == "__main__":
    res_dir = "res"
    rows = []
    for run in os.listdir(res_dir):
        for d in os.listdir(f"{res_dir}/{run}"):
            best_metrics = best_nprobs = None
            cputime = 0.0
            for i, p in enumerate(
                sorted(
                    os.listdir(f"{res_dir}/{run}/{d}"),
                    key=lambda x: int(x.split("_")[0]),
                )
            ):
                with open(os.path.join(res_dir, run, d, p, "metrics.json")) as f:
                    metrics = json.load(f)

                if best_metrics is None:
                    best_metrics = metrics
                    best_nprobs = i + 1
                else:
                    if (
                        metrics["precision"]["mean"] > best_metrics["precision"]["mean"]
                    ) or (metrics["recall"]["mean"] > best_metrics["recall"]["mean"]):
                        best_metrics = metrics
                        best_nprobs = i + 1

            rows.append(
                {
                    "Domain": d,
                    "#Instances": best_nprobs,
                    "Precision": best_metrics["precision"]["mean"],
                    "Recall": best_metrics["recall"]["mean"],
                    "CPU time": best_metrics["CPU time"],
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values(by="Domain")

        # save to excel
        output_file = "summary.xlsx"
        df.to_excel(output_file, index=False, float_format="%.2f")
