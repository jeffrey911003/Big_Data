from fastapi import FastAPI
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# 初始化 FastAPI 應用程式
app = FastAPI(title="網頁推薦系統 API", description="提供網頁興趣分數預測與推薦清單")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 專題演示用 "*" 即可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 1. 在伺服器啟動時，載入模型與最新特徵資料庫（純淨真實數據）
# ============================================================
print("正在載入冠軍模型與真實特徵庫...")
model = joblib.load('best_recommend_model_hgb.pkl')
features_list = joblib.load('model_features_2.pkl')

# 載入特徵資料庫，並將 pid 設為索引（此處 index 將完美保持 Page_XXXXXXXX 格式）
feature_store = pd.read_csv('feature_store_2.csv', index_col=0)
print("載入完成！伺服器已準備就緒，推薦結果將 100% 由直方圖梯度提升模型自動分析（保留 Page_ 前綴）。")


# ============================================================
# 2. 定義 API 路由 (Endpoints)
# ============================================================

@app.get("/")
def read_root():
    return {"status": "success", "message": "推薦系統 API 運作中"}


@app.get("/recommend")
def get_recommendations(top_k: int = 5):
    """
    【AI 自動化推薦核心】
    直接讀取去識別化特徵送入直方圖梯度提升模型，即時分析出分數最高的網頁。
    這裡傳出去的 page_id 會原汁原味包含 Page_ 前綴。
    """
    try:
        # 1. 讀取所有候選網頁特徵，確保特徵順序與訓練時一致
        candidate_features = feature_store[features_list]
        
        # 2. 將特徵送入模型進行預測 (取得分類為 1 的直方圖梯度提升真實預測機率)
        scores = model.predict_proba(candidate_features)[:, 1]
        
        # 3. 將分數與原本的網頁 ID (此處已是 Page_XXXXXXXX) 結合
        results_df = pd.DataFrame({
            'page_id': candidate_features.index,
            'score': scores
        })
        
        # 4. 根據模型預估的分數由高到低排序，自動挑出前 top_k 名最優秀的網頁
        top_pages = results_df.sort_values(by='score', ascending=False).head(top_k)
        
        # 5. 整理成 JSON 格式回傳給前端
        recommendations = []
        for _, row in top_pages.iterrows():
            recommendations.append({
                "page_id": str(row['page_id']),  # 輸出標準的 "Page_4651444678"
                "confidence_score": round(float(row['score']), 4)
            })
            
        return {
            "status": "success",
            "returned_results": len(recommendations),
            "data": recommendations
        }
    except Exception as e:
        return {"status": "error", "message": f"全自動推薦計算失敗: {str(e)}"}


@app.get("/predict/{page_id}")
def predict_single_page(page_id: str):
    """
    查詢特定單一網頁的真實推薦分數
    支援兩種查詢輸入：
    1. 帶前綴的完整 ID (例如: Page_4651444678)
    2. 純數字的 ID (例如: 4651444678) -> 系統會自動幫忙補上 Page_
    """
    try:
        target_id = page_id
        
        # 💡 核心優化：如果傳進來的是純數字，自動幫他穿上「皮」以匹配資料庫 index
        if not target_id.startswith("Page_"):
            target_id = f"Page_{page_id}"
            
        # 精準匹配特徵庫索引
        if target_id in feature_store.index:
            target_page = feature_store.loc[[target_id]]
        else:
            return {"status": "error", "message": f"在去識別化特徵庫中找不到該網頁代號 '{page_id}' (已嘗試匹配 {target_id})"}
            
        target_features = target_page[features_list]
        score = model.predict_proba(target_features)[:, 1][0]
        
        return {
            "status": "success",
            "page_id": target_id,  # 統一回傳標準的 Page_XXXXXXXX 格式
            "confidence_score": round(float(score), 4),
            "recommend": bool(score > 0.5)
        }
    except Exception as e:
        return {"status": "error", "message": f"單一網頁預測失敗: {str(e)}"}


# ============================================================
# 【企業未來對接專區】完美保留，並升級相容格式
# ============================================================
@app.post("/update_enterprise_data")
def update_enterprise_data(payload: dict):
    """
    未來若有真實企業提供含有「點擊量」、「停留時間」的資料，
    可動態對應到現有的特徵關鍵字 (sum / max)，達成即時數據注入。
    """
    try:
        incoming_pid = payload.get("page_id")
        clicks = float(payload.get("click_count_sum", 0))
        duration = float(payload.get("duration_max", 0))
        
        if not incoming_pid:
            return {"status": "error", "message": "缺乏必填欄位 page_id"}
            
        # 💡 同步優化：防錯機制，如果企業傳進來的 ID 沒帶 Page_，自動幫忙補上
        if not str(incoming_pid).startswith("Page_"):
            incoming_pid = f"Page_{incoming_pid}"
            
        new_features = {}
        for col in features_list:
            if 'sum' in col.lower() or 'click' in col.lower():
                new_features[col] = clicks
            elif 'max' in col.lower() or 'duration' in col.lower():
                new_features[col] = duration
            else:
                new_features[col] = 0.0
                
        feature_store.loc[incoming_pid] = pd.Series(new_features)
        return {
            "status": "success",
            "message": f"成功將企業真實數據對齊並注入特徵庫！網頁 ID: {incoming_pid}",
            "mapped_features_count": len(new_features)
        }
    except Exception as e:
        return {"status": "error", "message": f"處理企業資料失敗: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)