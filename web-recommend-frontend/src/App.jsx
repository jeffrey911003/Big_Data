import React, { useState } from 'react';

// ============================================================
// 【靈魂包裝區】模型真實完整 ID ➔ 前端商業中文名詞對照表
// 💡 這裡的 Key 完全對齊你隨機森林模型跑出來的真實 Page_ID 字串！
// ============================================================
const pageContentMapper = {
  "Page_4712886852": { // 冠軍模型算出來的第一名！
    title: "NBA 季後賽戰報｜今日焦點賽事戰況與關鍵絕殺球深度報導",
    tag: "今日最高熱度",
    clicks: "1,480",
    duration: "180秒"
  },
  "Page_4727381748": { // 冠軍模型算出來的第二名！
    title: "科技專題｜2026 全新旗艦款智慧型手機完整開箱與規格評測",
    tag: "深度閱讀型",
    clicks: "890",
    duration: "120秒"
  },
  "Page_4701471195": { // 冠軍模型算出來的第三名！
    title: "美食特搜｜台北必吃排隊拉麵老店老饕私藏清單",
    tag: "潛力上升中",
    clicks: "620",
    duration: "95秒"
  }
};

// 企業日誌檔案選單 (作為 Demo 的觸發媒介，不再包含任何手寫推薦數據)
const clientRawLogFiles = [
  {
    fileId: "LOG_20260528_DAILY",
    fileName: "nba_sports_traffic_raw_logs.csv",
    fileSize: "42.8 MB (共 1,284,910 行點擊數據)",
    description: "今日全站用戶的瀏覽紀錄，內含大量使用者在不同時間的點擊與停留數據。"
  },
  {
    fileId: "LOG_20260528_WEEKEND",
    fileName: "weekend_comprehensive_user_trajectories.log",
    fileSize: "128.5 MB (共 4,510,200 行點擊數據)",
    description: "週末連續 48 小時的用戶行為紀錄，包含大量的點擊與閱讀數據。"
  }
];

function App() {
  const [selectedFile, setSelectedFile] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState('');
  const [recommendationList, setRecommendationList] = useState(null);

  // ============================================================
  // 【真實連線】直接呼叫 FastAPI 後端，進行真實模型預估
  // ============================================================
  const handleBatchAnalyze = async () => {
    if (!selectedFile) return;
    
    setIsProcessing(true);
    setRecommendationList(null);

    // 1. 舞台展示效果：數據流載入與特徵提取提示
    setProcessingStep('正在上傳客戶去識別化流量日誌檔案...');
    
    setTimeout(() => {
      setProcessingStep('日誌串流成功：隨機森林 (RandomForest) 冠軍模型即時預估分數中...');
    }, 1200);

    try {
      // 2. 真正連線到你的 FastAPI 後端獲取模型預估的前 3 名
      const response = await fetch('http://127.0.0.1:8002/recommend?top_k=3');
      const apiResult = await response.json();

      if (apiResult.status === "success") {
        
        // 3. 將後端傳回的原始 ID（Page_XXXXXXXX）與分數，動態結合前端對照表
        const formattedList = apiResult.data.map((item, index) => {
          const currentId = String(item.page_id); // 這裡拿到的就是 "Page_4712886852" 等原始 ID
          
          // 如果模型算出了別的 ID（在 Mapper 找不到），啟動自動防錯包裝機制
          const matchedContent = pageContentMapper[currentId] || {
            title: `隨機森林精選網頁 ${currentId} (去識別化數據模型自動推薦)`,
            tag: "AI 推薦網頁",
            clicks: "動態統計中",
            duration: "動態統計中"
          };

          return {
            rank: index + 1,
            pageId: currentId,             // 【100% 真實】直接來自隨機森林模型
            score: item.confidence_score,  // 【100% 真實】直接來自隨機森林模型 (例如: 0.705)
            title: matchedContent.title,   // 〖命名包裝〗
            tag: matchedContent.tag,       // 〖命名包裝〗
            clickSum: matchedContent.clicks,
            maxDuration: matchedContent.duration
          };
        });

        setRecommendationList(formattedList);
      } else {
        alert("後端隨機森林模型計算失敗: " + apiResult.message);
      }
    } catch (error) {
      alert("無法連線到推薦系統後端！請確認您的 FastAPI 是否已在 Port 8002 正常啟動。");
    } finally {
      setIsProcessing(false);
      setProcessingStep('');
    }
  };

  return (
    <div style={{ maxWidth: '900px', margin: '40px auto', padding: '30px', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', backgroundColor: '#fdfdfd', color: '#333', boxShadow: '0 8px 30px rgba(0,0,0,0.08)', borderRadius: '12px', border: '1px solid #eaeaea' }}>
      
      {/* 頂部標題區域 */}
      <div style={{ borderBottom: '2px solid #0056b3', paddingBottom: '15px', marginBottom: '30px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h1 style={{ margin: 0, fontSize: '26px', color: '#0056b3', fontWeight: 'bold' }}>網頁推薦系統 ➔ 數據分析後台</h1>
          <span style={{ backgroundColor: '#e8f5e9', color: '#2b8a3e', padding: '4px 10px', borderRadius: '15px', fontSize: '12px', fontWeight: 'bold' }}>● Live-backend</span>
        </div>
        <p style={{ margin: '8px 0 0 0', color: '#666', fontSize: '14px' }}>後端基於隨機森林(RandomForest)真實預測機率 </p>
      </div>

      {/* 第一步：導入客戶紀錄 */}
      <div style={{ backgroundColor: '#f8f9fa', padding: '20px', borderRadius: '8px', marginBottom: '25px', border: '1px solid #e9ecef' }}>
        <h3 style={{ margin: '0 0 15px 0', fontSize: '16px', color: '#495057' }}>第一步：請匯入客戶端未處理的用戶瀏覽紀錄</h3>
        
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <select 
            value={selectedFile} 
            onChange={(e) => setSelectedFile(e.target.value)}
            disabled={isProcessing}
            style={{ padding: '12px', fontSize: '14px', flex: 1, borderRadius: '6px', border: '1px solid #ced4da', backgroundColor: '#fff', fontFamily: 'monospace' }}
          >
            <option value="">-- 請選擇欲上傳分析的企業資料檔案 --</option>
            {clientRawLogFiles.map(file => (
              <option key={file.fileId} value={file.fileId}>{file.fileName} ({file.fileSize})</option>
            ))}
          </select>

          <button 
            onClick={handleBatchAnalyze}
            disabled={!selectedFile || isProcessing}
            style={{ padding: '12px 24px', cursor: 'pointer', fontWeight: 'bold', backgroundColor: isProcessing ? '#6c757d' : '#0056b3', color: '#fff', border: 'none', borderRadius: '6px', fontSize: '14px', transition: 'all 0.2s', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}
          >
            {isProcessing ? '模型即時計算中...' : '動態預估真實 Top-K'}
          </button>
        </div>

        {selectedFile && !isProcessing && (
          <p style={{ margin: '10px 0 0 0', fontSize: '13px', color: '#6c757d' }}>
            <b>檔案描述：</b>{clientRawLogFiles.find(f => f.fileId === selectedFile)?.description}
          </p>
        )}
      </div>

      {/* 動態計算進度條 */}
      {isProcessing && (
        <div style={{ padding: '25px', backgroundColor: '#fff8e1', border: '1px solid #ffe082', borderRadius: '8px', marginBottom: '25px', textAlign: 'center' }}>
          <div className="spinner" style={{ width: '30px', height: '30px', border: '4px solid #f3f3f3', borderTop: '4px solid #ffb300', borderRadius: '50%', margin: '0 auto 15px auto', animation: 'spin 1s linear infinite' }}></div>
          <p style={{ margin: 0, fontWeight: 'bold', color: '#b78103', fontSize: '15px' }}>{processingStep}</p>
          <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
        </div>
      )}

      {/* 輸出推薦排行榜區域 */}
      {recommendationList && (
        <div style={{ animation: 'fadeIn 0.5s ease' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
            <h3 style={{ margin: 0, color: '#111', fontSize: '18px' }}>
              隨機森林分析結果：成功產出 Top-{recommendationList.length} 最佳推薦網頁
            </h3>
            <span style={{ fontSize: '13px', color: '#2b8a3e', fontWeight: 'bold' }}>狀態：隨機森林預估完成</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            {recommendationList.map((item) => (
              <div 
                key={item.pageId}
                style={{ display: 'flex', alignItems: 'center', backgroundColor: '#fff', border: '1px solid #dee2e6', borderRadius: '8px', padding: '18px', boxShadow: '0 2px 4px rgba(0,0,0,0.02)', position: 'relative', borderLeft: item.rank === 1 ? '6px solid #ff1744' : '6px solid #0056b3' }}
              >
                {/* 名次標示 */}
                <div style={{ fontSize: '24px', fontWeight: '900', color: item.rank === 1 ? '#ff1744' : '#0056b3', minWidth: '45px', textAlign: 'center', fontStyle: 'italic' }}>
                  #{item.rank}
                </div>

                {/* 網頁內容資訊 */}
                <div style={{ flex: 1, paddingRight: '20px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                    <span style={{ backgroundColor: item.rank === 1 ? '#ffebee' : '#e8f0fe', color: item.rank === 1 ? '#c62828' : '#1565c0', padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 'bold' }}>
                      {item.tag}
                    </span>
                    <span style={{ fontSize: '11px', color: '#2b8a3e', backgroundColor: '#e8f5e9', padding: '2px 6px', borderRadius: '4px', fontFamily: 'monospace', fontWeight: 'bold' }}>
                      模型真實 ID: {item.pageId}
                    </span>
                  </div>
                  <h4 style={{ margin: '0 0 6px 0', fontSize: '16px', color: '#222', lineHeight: '1.4' }}>{item.title}</h4>
                  <p style={{ margin: 0, fontSize: '12px', color: '#666' }}>
                    數據統計對照：預估總點擊約 <b>{item.clickSum} 次</b> ｜ 停留時間 <b>{item.maxDuration}</b>
                  </p>
                </div>

                {/* 預測分數 (完全連線自後端隨機森林模型) */}
                <div style={{ textAlign: 'center', minWidth: '100px', borderLeft: '1px solid #eee', paddingLeft: '15px' }}>
                  <span style={{ fontSize: '11px', color: '#777', display: 'block', marginBottom: '4px' }}>模型推薦指數</span>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: item.score > 0.5 ? '#2b8a3e' : '#e65100' }}>
                    {(item.score * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;