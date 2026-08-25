import os
import pandas as pd
import lsk_muufl
import lsk_weifang


def run_benchmark():
    # --- 实验配置矩阵 ---
    # 如果未来有新模型，只需在此列表中添加对应的模块和名称
    experiments = [
        {
            'dataset_name': 'weifang',
            'model_name': 'LSKNet',
            'module': lsk_weifang,
            'sample_gradients': [2, 4, 6, 8, 10],  # Weifang 标准为 10
            'report_file': 'LSKNet_Weifang_report.txt'
        },
        {
            'dataset_name': 'muufl',
            'model_name': 'LSKNet',
            'module': lsk_muufl,
            'sample_gradients': [2, 4, 6, 8, 10],  # 对应 Weifang 标准 10
            # 'sample_gradients': [5, 10, 15, 20],  # MUUFL 标准为 20
            'report_file': 'LSKNet_MUUFL_report.txt'
        }
    ]

    for exp in experiments:
        ds = exp['dataset_name']
        mn = exp['model_name']
        mod = exp['module']
        summary_data = []

        # 核心根目录
        base_dir = f"fewshot/{ds}/{mn}"

        for n in exp['sample_gradients']:
            print(f"\n[Benchmark] Dataset: {ds} | Model: {mn} | Samples: {n}")

            # --- 遵循规范的路径设置 ---
            # 动态修改源码中的全局路径变量
            mod.TRAIN_SAMPLES_PER_CLASS = n
            mod.SAVE_DIR_PARAMS = f"{base_dir}/samples_{n}/weights"
            mod.SAVE_DIR_RESULTS = f"{base_dir}/samples_{n}/reports"
            mod.SAVE_DIR_MAPS = f"{base_dir}/samples_{n}/maps"

            # 执行训练
            try:
                mod.main()
            except Exception as e:
                print(f"Error during {ds}_{mn}_n{n}: {e}")
                continue

            # --- 提取结果 ---
            report_path = f"{mod.SAVE_DIR_RESULTS}/{exp['report_file']}"
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # 从报告中通过关键词提取核心指标
                    oa = next(l for l in lines if "OA" in l).split(':')[-1].strip().replace('%', '')
                    aa = next(l for l in lines if "AA" in l).split(':')[-1].strip().replace('%', '')
                    kappa = next(l for l in lines if "Kappa" in l).split(':')[-1].strip().replace('%', '')

                    summary_data.append({
                        "Dataset": ds,
                        "Model": mn,
                        "Samples_Per_Class": n,
                        "OA (%)": oa,
                        "AA (%)": aa,
                        "Kappa": kappa
                    })

        # --- 生成该模型在该数据集下的标准汇总表 ---
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_filename = f"fewshot/{ds}/{ds}_{mn}_summary.csv"
            df.to_csv(summary_filename, index=False)
            print(f"\n[Finished] Summary saved to: {summary_filename}")


if __name__ == "__main__":
    run_benchmark()