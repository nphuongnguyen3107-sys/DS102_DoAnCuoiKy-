import os
import sys
import pandas as pd
import numpy as np

# Đảm bảo stdout/stderr sử dụng UTF-8 để tránh UnicodeEncodeError trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Thêm thư mục gốc của dự án vào sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

import ml_pipeline
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def main():
    X_train, X_test, y_train, y_test = ml_pipeline.load_data(
        x_path=os.path.join(project_dir, 'data/X.csv'), 
        y_path=os.path.join(project_dir, 'data/y.csv')
    )
    
    # Train all models with 50 trials (để khớp với báo cáo)
    print("Đang huấn luyện và tối ưu hóa các mô hình...")
    results = ml_pipeline.train_all_models(X_train, y_train, n_trials=50)
    
    print("\n" + "="*80)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP (361 mẫu)")
    print("="*80)
    
    test_results = []
    for name, key in [
        ("Logistic Regression", "lr"),
        ("XGBoost", "xgb"),
        ("LightGBM", "lgbm"),
        ("Random Forest", "rf"),
        ("Stacking", "stacking")
    ]:
        pipe, _, threshold = results[key]
        y_proba_test = pipe.predict_proba(X_test)[:, 1]
        y_pred_test = (y_proba_test >= threshold).astype(int)
        
        report = classification_report(y_test, y_pred_test, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_proba_test)
        
        test_results.append({
            "Mô hình": name,
            "Threshold": f"{threshold:.3f}",
            "Accuracy": f"{report['accuracy']*100:.2f}%",
            "Macro F1": f"{report['macro avg']['f1-score']*100:.2f}%",
            "Recall (Kháng - R)": f"{report['1']['recall']*100:.2f}%",
            "Precision (Kháng - R)": f"{report['1']['precision']*100:.2f}%",
            "ROC-AUC": f"{roc_auc:.4f}"
        })
        
    df_res = pd.DataFrame(test_results)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
