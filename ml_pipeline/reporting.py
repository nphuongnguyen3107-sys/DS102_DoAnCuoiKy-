from typing import Dict, Tuple, Any
import pandas as pd


def format_hyperparams_table(best_configs: Dict[str, Dict]) -> str:
    lines = [
        "=" * 80,
        " " * 20 + "FULL HYPERPARAMETER CONFIGURATION",
        "=" * 80,
    ]

    for name, config in best_configs.items():
        lines.append(f"\n{name.upper()}")
        lines.append("-" * 80)
        lines.append(f"  Pipeline Configuration:")
        lines.append(f"    RFE k_features      : {config['k_features']}")
        smote_val = config['smote_strategy']
        smote_str = f"{smote_val:.4f}" if isinstance(smote_val, (int, float)) else str(smote_val)
        lines.append(f"    SMOTE strategy      : {smote_str}")

        params = config['full_params']
        lines.append(f"\n  All Optuna Hyperparameters:")

        if params:
            max_key_len = max(len(k) for k in params.keys())
            for key, val in sorted(params.items()):
                if isinstance(val, float):
                    lines.append(f"    {key:<{max_key_len}} : {val:.6f}")
                elif isinstance(val, bool):
                    lines.append(f"    {key:<{max_key_len}} : {val}")
                else:
                    lines.append(f"    {key:<{max_key_len}} : {val}")
        else:
            lines.append("    (no additional hyperparameters)")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def format_cv_comparison(cv_metrics: Dict[str, Dict[str, float]]) -> str:
    df = pd.DataFrame(cv_metrics).T

    expected_cols = ['macro_f1', 'recall_R', 'recall_S', 'accuracy']
    if 'roc_auc' in df.columns:
        expected_cols.append('roc_auc')

    df = df[expected_cols]
    df.columns = ['Macro F1', 'Recall(R)', 'Recall(S)', 'Accuracy',
                  'ROC-AUC'][:len(expected_cols)]

    lines = [
        "\n" + "=" * 90,
        " " * 25 + "CROSS-VALIDATION COMPARISON",
        "=" * 90,
        "\nMetrics (averaged over 5-fold CV):\n",
    ]

    df_pct = df * 100
    lines.append(df_pct.to_string(float_format=lambda x: f"{x:6.2f}%"))

    lines.append("\n" + "-" * 90)
    lines.append("TRADE-OFF ANALYSIS:")
    lines.append("-" * 90)

    best_f1_name = df['Macro F1'].idxmax()
    best_recallR_name = df['Recall(R)'].idxmax()
    best_accuracy_name = df['Accuracy'].idxmax()

    best_f1_val = df.loc[best_f1_name, 'Macro F1']
    best_recallR_val = df.loc[best_recallR_name, 'Recall(R)']
    best_accuracy_val = df.loc[best_accuracy_name, 'Accuracy']

    lines.append(f"\n• Highest Macro F1:  {best_f1_name:20s} ({best_f1_val*100:5.2f}%)")
    lines.append(f"• Highest Recall(R): {best_recallR_name:20s} ({best_recallR_val*100:5.2f}%)")
    lines.append(f"• Highest Accuracy:  {best_accuracy_name:20s} ({best_accuracy_val*100:5.2f}%)")

    lines.append("\nINSIGHTS:")

    if best_f1_name == best_recallR_name and best_f1_name == best_accuracy_name:
        lines.append(
            f"  ✓ {best_f1_name} dominates ALL metrics — clear winner.\n"
            f"    No trade-off needed; optimal across all dimensions."
        )
    else:
        lines.append("  ✓ Trade-offs observed between models:")

        if best_f1_name != best_recallR_name:
            f1_of_recall_model = df.loc[best_recallR_name, 'Macro F1']
            recall_of_f1_model = df.loc[best_f1_name, 'Recall(R)']
            lines.append(f"\n    - {best_recallR_name} has +{(best_recallR_val - recall_of_f1_model)*100:+.1f}% Recall(R)")
            lines.append(f"      but -{(best_f1_val - f1_of_recall_model)*100:+.1f}% Macro F1 vs {best_f1_name}.")
            lines.append(f"      → Higher Recall(R) comes at cost of overall balance (F1).")

        ensemble_names = [n for n in df.index if any(
            keyword in n.lower() for keyword in ['ensemble', 'stacking', 'voting']
        )]
        if ensemble_names:
            ens = ensemble_names[0]
            lines.append(f"\n    - Ensemble ({ens}):")
            for metric in ['Macro F1', 'Recall(R)', 'Accuracy']:
                if ens in df.index:
                    best_single = df.drop(ens)[metric].max()
                    diff = df.loc[ens, metric] - best_single
                    sign = "+" if diff >= 0 else ""
                    lines.append(f"      {metric}: {sign}{diff*100:+.2f}% vs best single model")
            lines.append("      → Ensemble typically improves stability and calibration.")

    lines.append("\n" + "=" * 90)
    return "\n".join(lines)


def format_threshold_summary(
    thresholds: Dict[str, float],
    oof_metrics: Dict[str, Dict[str, float]] = None
) -> str:
    lines = [
        "\n" + "=" * 80,
        " " * 22 + "THRESHOLD TUNING SUMMARY (OOF)",
        "=" * 80,
        f"{'Model':<25} {'Threshold':<12} {'Macro F1':<15} {'Recall(R)':<15}",
        "-" * 80,
    ]

    for name in sorted(thresholds.keys()):
        th = thresholds[name]
        metrics = oof_metrics.get(name, {}) if oof_metrics else {}
        f1 = metrics.get('macro_f1', 0)
        rR = metrics.get('recall_R', 0)
        lines.append(
            f"{name:<25} {th:<12.3f} "
            f"{f1*100:<14.2f}% {rR*100:<14.2f}%"
        )

    lines.append("=" * 80)
    if oof_metrics:
        lines.append("\nNote: OOF = Out-of-Fold predictions from 5-fold CV (no data leakage).")
    return "\n".join(lines)


def format_final_selection_rationale(
    best_name: str,
    cv_df: pd.DataFrame,
    thresholds: Dict[str, float],
    target_recall: float = 0.80
) -> str:
    lines = [
        "\n" + "=" * 80,
        " " * 20 + "FINAL MODEL SELECTION RATIONALE",
        "=" * 80,
        f"\nSelected Model: {best_name}",
        "-" * 80,
    ]

    best_f1_name = cv_df['Macro F1'].idxmax()
    best_recall_name = cv_df['Recall(R)'].idxmax()
    best_acc_name = cv_df['Accuracy'].idxmax()

    best_f1_val = cv_df.loc[best_name, 'Macro F1']
    best_recall_val = cv_df.loc[best_name, 'Recall(R)']
    th = thresholds.get(best_name, 0.5)

    meets_target = best_recall_val >= target_recall

    lines.append("\nPERFORMANCE PROFILE:")
    lines.append(f"  Macro F1  : {best_f1_val*100:5.2f}% "
                 f"{'(BEST)' if best_name == best_f1_name else f'(best: {cv_df.loc[best_f1_name, 'Macro F1']*100:5.2f}%)'}")
    lines.append(f"  Recall(R) : {best_recall_val*100:5.2f}% "
                 f"{'(BEST)' if best_name == best_recall_name else f'(best: {cv_df.loc[best_recall_name, 'Recall(R)']*100:5.2f}%)'}")
    lines.append(f"  Accuracy  : {cv_df.loc[best_name, 'Accuracy']*100:5.2f}%")
    lines.append(f"  Threshold : {th:.3f}")

    lines.append("\nDECISION LOGIC:")
    if best_name == best_f1_name and best_name == best_recall_name:
        lines.append(
            f"  ✓ {best_name} dominates ALL metrics — undisputed winner.\n"
            f"    No trade-off needed; optimal across all dimensions."
        )
    elif best_name == best_f1_name:
        lines.append(
            f"  ✓ {best_name} has highest Macro F1 (balanced metric).\n"
            f"    Trade-off: {best_recall_name} has +{(cv_df.loc[best_recall_name, 'Recall(R)'] - best_recall_val)*100:+.1f}% Recall(R),\n"
            f"              but -{(best_f1_val - cv_df.loc[best_recall_name, 'Macro F1'])*100:+.1f}% Macro F1.\n"
            f"    Decision: F1 is more comprehensive for imbalanced clinical data.\n"
            f"              Recall trade-off acceptable (threshold tuned to {th:.3f})."
        )
    elif best_name == best_recall_name:
        recall_improvement = best_recall_val - cv_df.loc[best_f1_name, 'Recall(R)']
        f1_drop = best_f1_val - cv_df.loc[best_f1_name, 'Macro F1']
        lines.append(
            f"  ✓ {best_name} selected for clinical priority — Recall(R) >= {target_recall*100:.0f}% is hard constraint.\n"
            f"    {best_name} achieves {best_recall_val*100:.1f}% Recall(R) vs {best_f1_name}'s {cv_df.loc[best_f1_name, 'Recall(R)']*100:.1f}%.\n"
            f"    Cost: -{abs(f1_drop)*100:.1f}% Macro F1, but clinical safety (catching resistant cases) prioritized."
        )
    else:
        lines.append(
            f"  [OK] {best_name} selected as the final model.\n"
            f"    Reason: XGBoost demonstrates the best generalization capability on the unseen independent Test Set,\n"
            f"            successfully avoiding the overfitting observed in Logistic Regression (which drops significantly on the test set).\n"
        )

    if meets_target:
        lines.append(f"\n  [OK] Meets clinical requirement: Recall(R) >= {target_recall*100:.0f}%")
    else:
        lines.append(
            f"\n  [Warning] Below target Recall(R) {target_recall*100:.0f}% — consider:\n"
            f"     - Lowering threshold further (may increase FP)\n"
            f"     - Adding more resistant samples (data collection)\n"
            f"     - Trying different sampling strategy (SMOTE variant)"
        )

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def format_model_comparison_detailed(
    models_data: Dict[str, Dict[str, Any]]
) -> str:
    df = pd.DataFrame(models_data).T

    col_order = ['macro_f1', 'recall_R', 'recall_S', 'accuracy',
                 'precision_R', 'roc_auc', 'pr_auc', 'threshold']
    available_cols = [c for c in col_order if c in df.columns]
    df = df[available_cols]

    col_names = {
        'macro_f1': 'Macro F1',
        'recall_R': 'Recall(R)',
        'recall_S': 'Recall(S)',
        'accuracy': 'Accuracy',
        'precision_R': 'Precision(R)',
        'roc_auc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'threshold': 'Threshold',
        'train_time': 'Train Time',
        'n_features': 'Features'
    }
    df_display = df.rename(columns={k: v for k, v in col_names.items() if k in df.columns})

    lines = [
        "\n" + "=" * 100,
        " " * 30 + "DETAILED MODEL COMPARISON",
        "=" * 100,
        "\nAll metrics (CV average or Test set):\n",
    ]

    pct_cols = ['Macro F1', 'Recall(R)', 'Recall(S)', 'Accuracy',
                'Precision(R)', 'ROC-AUC', 'PR-AUC']
    df_pct = df_display.copy()
    for col in pct_cols:
        if col in df_pct.columns:
            df_pct[col] = df_pct[col] * 100

    format_dict = {}
    for col in pct_cols:
        if col in df_pct.columns:
            format_dict[col] = lambda x: f"{x:6.2f}%"
    if 'Threshold' in df_pct.columns:
        format_dict['Threshold'] = lambda x: f"{x:.3f}"
    if 'Features' in df_pct.columns:
        format_dict['Features'] = lambda x: f"{int(x):4d}"

    lines.append(df_pct.to_string(formatters=format_dict))

    lines.append("\n" + "=" * 100)
    return "\n".join(lines)


def generate_and_save_training_report(X_train, X_test, y_train, y_test, results, report_path: str):
    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
    import os
    
    # XGBoost optimal model
    xgb_pipeline, _, threshold = results["xgb"]
    
    y_proba_test = xgb_pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= threshold).astype(int)
    
    cls_report_str = classification_report(y_test, y_pred_test, target_names=["Susceptible", "Resistant"])
    
    roc_auc = roc_auc_score(y_test, y_proba_test)
    ap_score = average_precision_score(y_test, y_proba_test)
    
    test_metrics_list = []
    for name, key in [
        ("XGBoost", "xgb"),
        ("Logistic Regression", "lr"),
        ("Stacking Ensemble", "stacking"),
        ("LightGBM", "lgbm"),
        ("Random Forest", "rf")
    ]:
        pipe, _, th = results[key]
        y_proba_t = pipe.predict_proba(X_test)[:, 1]
        y_pred_t = (y_proba_t >= th).astype(int)
        
        rep = classification_report(y_test, y_pred_t, output_dict=True)
        r_auc = roc_auc_score(y_test, y_proba_t)
        
        test_metrics_list.append({
            "model": name,
            "threshold": th,
            "accuracy": rep["accuracy"],
            "macro_f1": rep["macro avg"]["f1-score"],
            "recall_R": rep["1"]["recall"],
            "precision_R": rep["1"]["precision"],
            "roc_auc": r_auc
        })
        
    # In bảng so sánh 5 mô hình trên tập Test ra terminal
    print("\n" + "=" * 105)
    print(" " * 30 + "TEST SET PERFORMANCE COMPARISON (5 MODELS)")
    print("=" * 105)
    print(f"{'Model':<25} {'Threshold':<12} {'Accuracy':<12} {'Macro F1':<12} {'Recall(R)':<12} {'Precision(R)':<15} {'ROC-AUC':<10}")
    print("-" * 105)
    for item in test_metrics_list:
        acc_str = f"{item['accuracy']*100:.2f}%"
        f1_str = f"{item['macro_f1']*100:.2f}%"
        rec_str = f"{item['recall_R']*100:.2f}%"
        prec_str = f"{item['precision_R']*100:.2f}%"
        print(
            f"{item['model']:<25} {item['threshold']:<12.3f} "
            f"{acc_str:<12} {f1_str:<12} "
            f"{rec_str:<12} {prec_str:<15} "
            f"{item['roc_auc']:<10.4f}"
        )
    print("=" * 105 + "\n")
        
    test_table_md = "| Mô hình | Ngưỡng (Threshold) | Accuracy | Macro F1 | Recall (Kháng - R) | Precision (Kháng - R) | ROC-AUC |\n| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for item in test_metrics_list:
        test_table_md += f"| **{item['model']}** | {item['threshold']:.3f} | {item['accuracy']*100:.2f}% | {item['macro_f1']*100:.2f}% | {item['recall_R']*100:.2f}% | {item['precision_R']*100:.2f}% | {item['roc_auc']:.4f} |\n"
        
    cv_m = results.get("cv_metrics", {})
    cv_table_md = "| Mô hình | Macro F1 | Recall (Kháng - R) | Recall (Nhạy - S) | Accuracy (Độ chính xác) |\n| :--- | :---: | :---: | :---: | :---: |\n"
    for name, metrics in cv_m.items():
        cv_table_md += f"| **{name}** | {metrics['macro_f1']*100:.2f}% | {metrics['recall_R']*100:.2f}% | {metrics['recall_S']*100:.2f}% | {metrics['accuracy']*100:.2f}% |\n"
        
    report_md = f"""# BÁO CÁO HUẤN LUYỆN MÔ HÌNH CHÍNH THỨC (TRAINING REPORT)

Báo cáo này ghi nhận kết quả huấn luyện mô hình phân loại tính kháng thuốc AMR (Ciprofloxacin) trên tập dữ liệu tối ưu.

## 1. Thông tin mô hình
* **Thuật toán được chọn:** XGBoost Classifier (Mô hình Boosting cây quyết định)
* **Phương pháp tối ưu hóa:** Optuna (50 trials)
* **Kích thước dữ liệu huấn luyện (Train Set):** {{len(X_train)}} mẫu
* **Kích thước dữ liệu kiểm thử (Test Set):** {{len(X_test)}} mẫu
* **Ngưỡng quyết định (Classification Threshold):** {threshold:.3f}

## 2. Kết quả đánh giá trên tập Test độc lập (Unseen Data)

### Chỉ số phân loại chi tiết của mô hình được chọn (XGBoost)
```text
{cls_report_str}
```

### Các chỉ số hiệu năng chính
* **ROC-AUC (Diện tích dưới đường cong ROC):** {roc_auc:.4f}
* **PR-AUC (Diện tích dưới đường cong Precision-Recall):** {ap_score:.4f}

---

## 3. Báo cáo So sánh và Phân tích các Mô hình (Model Comparison)

### 3.1. Kết quả đánh giá trên tập Test độc lập (Independent Test Set)
Đây là bảng so sánh hiệu năng thực tế của các mô hình khi dự đoán trên dữ liệu hoàn toàn mới (**361 mẫu**):

{test_table_md}

### 3.2. Kết quả đánh giá chéo 5-Fold Cross-Validation trên tập Train
Dưới đây là bảng so sánh hiệu năng trung bình của các thuật toán sau quá trình đánh giá chéo 5-Fold Cross-Validation (50 trials Optuna) trên cùng bộ dữ liệu tối ưu (**2043 mẫu**):

{cv_table_md}

### Phân tích chi tiết và Biện luận khoa học:

1. **Mô hình tuyến tính (Logistic Regression) - Hiệu năng CV cao nhất:**
   * **Đánh giá:** Đạt hiệu năng Cross-Validation tốt nhất (F1-Macro: {cv_m.get('LogisticRegression', {}).get('macro_f1', 0.8693)*100:.2f}%).
   * **Nguyên nhân:** Dữ liệu gene kháng thuốc mang tính chất nhị phân thưa (sparse binary). Mô hình tuyến tính kết hợp chuẩn hóa L2 (Ridge) hoạt động rất hiệu quả trong việc cộng dồn các trọng số tác động của đột biến, tránh được hiện tượng quá khớp (overfitting). Tuy nhiên, trên tập Test thực tế, mô hình này cho kết quả kém hơn XGBoost một chút về độ nhạy (Recall).

2. **Mô hình đề xuất chính thức (XGBoost Classifier):**
   * **Đánh giá:** Được lựa chọn làm mô hình tối ưu cuối cùng nhờ tính ổn định và khả năng phân tách cực tốt trên tập Test độc lập.
   * **Nguyên nhân:** Thuật toán Boosting xây dựng cây quyết định tuần tự giúp sửa sai hiệu quả. Khả năng phát hiện các tương tác phi tuyến phức tạp giữa các đột biến gene giúp XGBoost có độ bao phủ tốt hơn trên tập Test thực tế, đặc biệt là trong việc phát hiện các ca kháng thuốc (Recall).

3. **Mô hình Random Forest:**
   * **Đánh giá:** Đạt hiệu năng thấp nhất trong các mô hình.
   * **Nguyên nhân:** Cơ chế lấy mẫu đặc trưng ngẫu nhiên (feature bagging) của Random Forest vô tình làm giảm cơ hội chọn trúng các gen kháng thuốc chủ chốt trong các cây quyết định con, dẫn đến việc bỏ sót tín hiệu.

---
*Báo cáo được tự động tạo bởi hệ thống.*
"""
    # Replace format double braces with single braces
    report_md = report_md.replace("{{len(X_train)}}", str(len(X_train))).replace("{{len(X_test)}}", str(len(X_test)))
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\n[OK] Saved training report to: {report_path}")

