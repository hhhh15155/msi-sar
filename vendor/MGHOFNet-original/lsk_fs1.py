import os
import pandas as pd
import numpy as np
import lsk_muufl
import lsk_weifang


def run_benchmark():
    # --- 实验配置矩阵 ---
    experiments = [
        {
            'dataset_name': 'weifang',
            'model_name': 'LSKNet',
            'module': lsk_weifang,
            'sample_gradients': [2, 4, 6, 8, 10],
            'report_file': 'LSKNet_Weifang_report.txt'
        },
        {
            'dataset_name': 'muufl',
            'model_name': 'LSKNet',
            'module': lsk_muufl,
            'sample_gradients': [2, 4, 6, 8, 10],
            'report_file': 'LSKNet_MUUFL_report.txt'
        }
    ]

    for exp in experiments:
        ds = exp['dataset_name']
        mn = exp['model_name']
        mod = exp['module']
        summary_data = []
        base_dir = f"fewshot1/{ds}/{mn}"

        for n in exp['sample_gradients']:
            # 用于存储当前 n 下 5 次运行的指标
            run_metrics = {"OA": [], "AA": [], "Kappa": []}

            # --- 核心改动：运行 5 次并选用不同 seed ---
            seeds = [42, 43, 44, 45, 46]
            for run_idx, seed in enumerate(seeds):
                print(f"\n[Benchmark] Dataset: {ds} | Samples: {n} | Run: {run_idx + 1}/5 (Seed: {seed})")

                # 动态设置参数
                mod.RANDOM_SEED = seed
                mod.TRAIN_SAMPLES_PER_CLASS = n
                # 为了防止结果覆盖，建议路径区分 run，但此处严格遵循你“不改动其他”的要求
                mod.SAVE_DIR_PARAMS = f"{base_dir}/samples_{n}/run_{run_idx}/weights"
                mod.SAVE_DIR_RESULTS = f"{base_dir}/samples_{n}/run_{run_idx}/reports"
                mod.SAVE_DIR_MAPS = f"{base_dir}/samples_{n}/run_{run_idx}/maps"

                try:
                    mod.main()
                except Exception as e:
                    print(f"Error during {ds}_n{n}_run{run_idx}: {e}")
                    continue

                # 提取单次运行结果
                report_path = f"{mod.SAVE_DIR_RESULTS}/{exp['report_file']}"
                if os.path.exists(report_path):
                    with open(report_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        oa = float(next(l for l in lines if "OA" in l).split(':')[-1].strip().replace('%', ''))
                        aa = float(next(l for l in lines if "AA" in l).split(':')[-1].strip().replace('%', ''))
                        kappa = float(next(l for l in lines if "Kappa" in l).split(':')[-1].strip().replace('%', ''))

                        run_metrics["OA"].append(oa)
                        run_metrics["AA"].append(aa)
                        run_metrics["Kappa"].append(kappa)

            # 计算 5 次实验的均值和标准差
            if run_metrics["OA"]:
                summary_data.append({
                    "Dataset": ds,
                    "Samples_Per_Class": n,
                    "OA_Mean": np.mean(run_metrics["OA"]),
                    "OA_Std": np.std(run_metrics["OA"]),
                    "AA_Mean": np.mean(run_metrics["AA"]),
                    "AA_Std": np.std(run_metrics["AA"]),
                    "Kappa_Mean": np.mean(run_metrics["Kappa"]),
                    "Kappa_Std": np.std(run_metrics["Kappa"])
                })

        # 生成汇总表
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_filename = f"fewshot1/{ds}/{ds}_{mn}_summary_stat.csv"
            df.to_csv(summary_filename, index=False)
            print(f"\n[Finished] Statistical summary saved to: {summary_filename}")


if __name__ == "__main__":
    run_benchmark()