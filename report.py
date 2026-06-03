import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 階段 1：載入資料與準備
# ==========================================
print("階段 1：載入資料")
train_path = "web_recom_train.csv"
test_path = "web_recom_test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
features = [f for f in train_df.columns if f not in ['pid', 'y']]

all_experiment_results = []



# =====================================================================
# 階段 2：萬用 Pooling 聚合函式 (實作架構圖 Step 3 的 5 種方法)
# =====================================================================
def get_bag_aggregation(X_data, pids, ys=None, agg_type='sum'):
    if isinstance(X_data, np.ndarray):
        X_data = pd.DataFrame(X_data, columns=features)
        
    temp_df = X_data.copy()
    temp_df['pid'] = pids.values
    
    if ys is not None: 
        temp_df['y'] = ys.values
        
    if agg_type == 'sum':
        bag_df = temp_df.groupby('pid').sum()
    elif agg_type == 'max':
        bag_df = temp_df.groupby('pid').max()
    elif agg_type == 'mean':
        bag_df = temp_df.groupby('pid').mean()
    elif agg_type == 'nonzero-count':
        feats = [c for c in temp_df.columns if c not in ['pid', 'y']]
        bag_df_feats = (temp_df[feats] > 0).astype(int)
        bag_df_feats['pid'] = temp_df['pid']
        bag_df = bag_df_feats.groupby('pid').sum()
        if ys is not None:
            bag_df['y'] = temp_df.groupby('pid')['y'].max()
    elif agg_type == 'bag_size':
        bag_df = pd.DataFrame(temp_df.groupby('pid').size(), columns=['bag_size'])
        if ys is not None:
            bag_df['y'] = temp_df.groupby('pid')['y'].max()
    else:
        raise ValueError("不支援的 agg_type！")

    if ys is not None and 'y' in bag_df.columns: 
        y_out = (bag_df['y'] > 0).astype(int)
        return bag_df.drop(columns=['y']), y_out
    
    return bag_df, None

# ==========================================================
# 階段 3：觀察單一網頁在 5 種聚合方式下的數值差異
# ==========================================================
print("\n階段 3：萃取並觀察 5 種聚合特徵...")
X_tr_raw = train_df[features].copy()
X_te_raw = test_df[features].copy()

bag_sum, _ = get_bag_aggregation(X_tr_raw, train_df['pid'], train_df['y'], agg_type='sum')
bag_max, _ = get_bag_aggregation(X_tr_raw, train_df['pid'], train_df['y'], agg_type='max')
bag_mean, _ = get_bag_aggregation(X_tr_raw, train_df['pid'], train_df['y'], agg_type='mean')
bag_nz, _ = get_bag_aggregation(X_tr_raw, train_df['pid'], train_df['y'], agg_type='nonzero-count')
bag_size, _ = get_bag_aggregation(X_tr_raw, train_df['pid'], train_df['y'], agg_type='bag_size')

sample_pid = train_df['pid'].unique()[0]
print(f"\n🎯 觀察目標網頁 ID: {sample_pid}")
print(f"📦 這個網頁的 Bag Size (日誌總筆數): {bag_size.loc[sample_pid, 'bag_size']} 筆")

comparison_df = pd.DataFrame({
    'Sum (總和)': bag_sum.loc[sample_pid],
    'Max (最大值)': bag_max.loc[sample_pid],
    'Mean (平均值)': bag_mean.loc[sample_pid],
    'Nonzero-count (非零次)': bag_nz.loc[sample_pid]
})
active_features = comparison_df[comparison_df['Sum (總和)'] > 0].head(15)

print("\n📊 各種聚合方式的實際數值比較表：")
# 使用 .to_string() 避免 tabulate 套件未安裝的報錯
print(active_features.to_string())

# 計算總共有幾個包 (不重複的 pid 數量)
train_bags = train_df['pid'].nunique()
test_bags = test_df['pid'].nunique()

print(f"📦 訓練集總共分成了：{train_bags} 個包")
print(f"📦 測試集總共分成了：{test_bags} 個包")

# ==========================================================
# 階段 4：執行模型與特徵實驗 (先聚合、再前處理)
# ==========================================================
print("\n階段 4：開始執行所有模型與特徵實驗")

# 實驗 A：5 種 Representation
X_train_bag_raw, y_train_bag = get_bag_aggregation(X_tr_raw, train_df['pid'], train_df['y'], agg_type='max')
X_test_bag_raw, y_test_bag = get_bag_aggregation(X_te_raw, test_df['pid'], test_df['y'], agg_type='max')

methods = ['Raw', 'Log', 'TF-IDF', 'Binary', 'PCA_SVD']
best_binary_X_train, best_binary_X_test = None, None 
best_binary_y_train, best_binary_y_test = None, None

for method in methods:
    if method == 'Raw': 
        X_tr, X_te = X_train_bag_raw.copy(), X_test_bag_raw.copy()
    elif method == 'Log': 
        X_tr, X_te = np.log1p(X_train_bag_raw), np.log1p(X_test_bag_raw)
    elif method == 'Binary': 
        X_tr, X_te = (X_train_bag_raw > 0).astype(int), (X_test_bag_raw > 0).astype(int)
    elif method == 'TF-IDF':
        tfidf = TfidfTransformer()
        X_tr = tfidf.fit_transform(X_train_bag_raw).toarray()
        X_te = tfidf.transform(X_test_bag_raw).toarray()
    elif method == 'PCA_SVD':
        tfidf_svd = TfidfTransformer()
        svd = TruncatedSVD(n_components=100, random_state=42)
        X_tr_tfidf = tfidf_svd.fit_transform(X_train_bag_raw)
        X_te_tfidf = tfidf_svd.transform(X_test_bag_raw)
        X_tr = svd.fit_transform(X_tr_tfidf)
        X_te = svd.transform(X_te_tfidf)
        X_tr = pd.DataFrame(X_tr, index=X_train_bag_raw.index)
        X_te = pd.DataFrame(X_te, index=X_test_bag_raw.index)

    if method == 'Binary':
        best_binary_X_train, best_binary_y_train = X_tr, y_train_bag
        best_binary_X_test, best_binary_y_test = X_te, y_test_bag

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_tr, y_train_bag)
    auc_val = roc_auc_score(y_test_bag, clf.predict_proba(X_te)[:, 1])
    all_experiment_results.append({"Model": "Random Forest", "Feature": f"MaxPool + {method}", "AUC": auc_val})
    print(f"完成：RF + MaxPool + {method} | AUC = {auc_val:.4f}")

# 實驗 B：XGBoost
xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb_clf.fit(best_binary_X_train, best_binary_y_train)
xgb_auc = roc_auc_score(best_binary_y_test, xgb_clf.predict_proba(best_binary_X_test)[:, 1])
all_experiment_results.append({"Model": "XGBoost", "Feature": "MaxPool + Binary", "AUC": xgb_auc})
print(f"完成：XGBoost + MaxPool + Binary | AUC = {xgb_auc:.4f}")

# 實驗 C：優化版 (Sum Pooling + TF-IDF)
X_train_sum_raw, y_train_sum = get_bag_aggregation(X_tr_raw, train_df['pid'], train_df['y'], agg_type='sum')
X_test_sum_raw, y_test_sum = get_bag_aggregation(X_te_raw, test_df['pid'], test_df['y'], agg_type='sum')

tfidf_opt = TfidfTransformer()
X_tr_tfidf_sum = tfidf_opt.fit_transform(X_train_sum_raw).toarray()
X_te_tfidf_sum = tfidf_opt.transform(X_test_sum_raw).toarray()

rf_opt = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42)
rf_opt.fit(X_tr_tfidf_sum, y_train_sum)
opt_auc = roc_auc_score(y_test_sum, rf_opt.predict_proba(X_te_tfidf_sum)[:, 1])
all_experiment_results.append({"Model": "RF (Balanced)", "Feature": "SumPool + TF-IDF", "AUC": opt_auc})
print(f"完成：RF(Balanced) + SumPool + TF-IDF | AUC = {opt_auc:.4f}")

# 實驗 D：LightGBM
lgb_clf = lgb.LGBMClassifier(n_estimators=100, max_depth=10, random_state=42, verbose=-1)
lgb_clf.fit(best_binary_X_train, best_binary_y_train)
lgb_auc = roc_auc_score(best_binary_y_test, lgb_clf.predict_proba(best_binary_X_test)[:, 1])
all_experiment_results.append({"Model": "LightGBM", "Feature": "MaxPool + Binary", "AUC": lgb_auc})
print(f"完成：LightGBM + MaxPool + Binary | AUC = {lgb_auc:.4f}")

# 實驗 E：L1 正規化特徵篩選 (Logistic Regression)
lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
lr_l1.fit(best_binary_X_train, best_binary_y_train)
lr_auc = roc_auc_score(best_binary_y_test, lr_l1.predict_proba(best_binary_X_test)[:, 1])
all_experiment_results.append({"Model": "LogReg (L1)", "Feature": "MaxPool + Binary", "AUC": lr_auc})
total_feats = best_binary_X_train.shape[1]
used_feats = np.sum(lr_l1.coef_ != 0)
print(f"完成：LogReg (L1) + MaxPool + Binary | AUC = {lr_auc:.4f} (維度 {total_feats} -> {used_feats})")

# 實驗 F：加入網格搜索最佳參數的 RandomForest
tuned_rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=2, random_state=42)
tuned_rf.fit(best_binary_X_train, best_binary_y_train)
tuned_auc = roc_auc_score(best_binary_y_test, tuned_rf.predict_proba(best_binary_X_test)[:, 1])
all_experiment_results.append({"Model": "RF (Tuned)", "Feature": "MaxPool + Binary", "AUC": tuned_auc})
print(f"完成：RF (最佳參數) + MaxPool + Binary | AUC = {tuned_auc:.4f}")

# ==========================================
# 階段 5：總結報表與視覺化圖表
# ==========================================
print("\n階段 5：總結報表與視覺化圖表")
report_df = pd.DataFrame(all_experiment_results)
report_df['Experiment_Name'] = report_df['Model'] + "\n(" + report_df['Feature'] + ")"
report_df = report_df.sort_values(by='AUC', ascending=False).reset_index(drop=True)

plt.figure(figsize=(16, 8))
sns.set_style("whitegrid")
barplot = sns.barplot(x='Experiment_Name', y='AUC', data=report_df, palette='viridis')
plt.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Random Guess Baseline (0.5)')
plt.ylim(0, 1.05) 
plt.title('Final Model Performance Comparison (AUC Score)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Algorithm & Feature Engineering Method', fontsize=12)
plt.ylabel('AUC Score', fontsize=12)

for i, v in enumerate(report_df['AUC']):
    plt.text(i, v + 0.015, f"{v:.4f}", ha='center', color='black', fontweight='bold', fontsize=10)

plt.legend(loc='upper right')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# ==========================================
# 階段 6：最佳模型排行榜與推薦清單 (補回區塊)
# ==========================================
print("\n模型健康度檢查：")

train_probs = tuned_rf.predict_proba(best_binary_X_train)[:, 1]
test_probs = tuned_rf.predict_proba(best_binary_X_test)[:, 1]

print(f"訓練集 AUC: {roc_auc_score(best_binary_y_train, train_probs):.4f}")
print(f"測試集 AUC: {roc_auc_score(best_binary_y_test, test_probs):.4f}")

print("\n模型預測分數排行榜")

default_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
default_rf.fit(best_binary_X_train, best_binary_y_train)

rf_probs = default_rf.predict_proba(best_binary_X_test)[:, 1]
rf_recommend_df = pd.DataFrame({
    'Page_ID': best_binary_X_test.index,
    'Actual_Interest': best_binary_y_test.values,
    'Score_Original_RF': rf_probs
}).sort_values(by='Score_Original_RF', ascending=False)

lgb_probs = lgb_clf.predict_proba(best_binary_X_test)[:, 1]
lgb_recommend_df = pd.DataFrame({
    'Page_ID': best_binary_X_test.index,
    'Actual_Interest': best_binary_y_test.values,
    'Score_LightGBM': lgb_probs
}).sort_values(by='Score_LightGBM', ascending=False)

tuned_recommend_df = pd.DataFrame({
    'Page_ID': best_binary_X_test.index,
    'Actual_Interest': best_binary_y_test.values,
    'Score_Tuned_RF': test_probs 
}).sort_values(by='Score_Tuned_RF', ascending=False)

print("\n[1] 原本的 Random Forest (預設參數) 推薦清單：")
print(rf_recommend_df.head(10).to_string(index=False))

print("\n[2] LightGBM 推薦清單：")
print(lgb_recommend_df.head(10).to_string(index=False))

print("\n[3] 調校過的 Random Forest (Tuned) 推薦清單：")
print(tuned_recommend_df.head(10).to_string(index=False))

# ==========================================
# 階段 7：PR-AUC 與混淆矩陣
# ==========================================
print("\n階段 7：模型準確度評估 (PR-AUC & 混淆矩陣)")

num_ones = sum(best_binary_y_test == 1)
num_zeros = sum(best_binary_y_test == 0)
print(f"測試集狀況：共有 {len(best_binary_y_test)} 個母網頁。")
print(f"其中有興趣的樣本數量為 {num_ones} 個，沒興趣的樣本數量為 {num_zeros} 個。")

precision, recall, _ = precision_recall_curve(best_binary_y_test, lgb_probs)
pr_auc_score = auc(recall, precision)

print("\n模型指標分數 (以 LightGBM 為例)：")
print(f" - ROC-AUC: {lgb_auc:.4f}")
print(f" - PR-AUC : {pr_auc_score:.4f}")

lgb_preds = (lgb_probs > 0.5).astype(int)
cm = confusion_matrix(best_binary_y_test, lgb_preds)

print("\n混淆矩陣 (Threshold = 0.5)：")
print("                  [模型預測 沒興趣]  [模型預測 有興趣]")
print(f"[實際 沒興趣 (0)]        {cm[0,0]}                 {cm[0,1]}")
print(f"[實際 有興趣 (1)]        {cm[1,0]}                 {cm[1,1]}")

# ==========================================
# 階段 8：儲存系統整合所需檔案 (供後端讀取)
# ==========================================
print("\n階段 8：儲存推薦系統檔案")

joblib.dump(lgb_clf, 'best_recommendation_model_lgb.pkl')
print(f"📦 已儲存模型: best_recommendation_model_lgb.pkl")

feature_list = list(best_binary_X_train.columns) if isinstance(best_binary_X_train, pd.DataFrame) else [f"Feature_{i}" for i in range(best_binary_X_train.shape[1])]
joblib.dump(feature_list, 'model_features.pkl')
print(f"📦 已儲存特徵清單: model_features.pkl")

feature_store_df = best_binary_X_test.copy()
if not isinstance(feature_store_df, pd.DataFrame):
    feature_store_df = pd.DataFrame(feature_store_df, index=best_binary_y_test.index, columns=feature_list)
feature_store_df.to_csv('feature_store.csv', index=True)
print(f"📦 已儲存特徵庫: feature_store.csv")

print("\n🎉 系統執行完畢！你可以啟動 app.py 與前端 React 應用程式了！")