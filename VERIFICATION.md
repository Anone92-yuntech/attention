# 雙肩選人／OC-SORT 整合驗證

日期：2026-10-27（Asia/Taipei）。所有產物位於本專案 `logs/tracking_verification/`。

## 自動測試

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -q
```

**149 項通過、0 失敗、0 跳過**。涵蓋來源肩寬選人、真正 OC-SORT 配對／預測／遺失、追蹤及恢復流程、
人物偵測原圖座標，以及本次的裁切手部回映、跨 ID／代次清除、未知狀態、窗口隔離、暫停、重選與錯誤清理。
來源測試預設保留原姿態門檻；`attention_app` 明確關閉無關趴桌關鍵點門檻，runner 測試及真實片段驗證該設定。
紀錄：`unittest.log`。`pip check` 另回報 `No broken requirements found`。

## 真實影片前 12 秒

```powershell
.\.venv\Scripts\python.exe -u attention.py --headless --max-frames 360 --output-csv logs/tracking_verification/prefix_actions.csv --report logs/tracking_verification/prefix_report.json
```

| 指標 | 結果 |
| --- | --- |
| 原影片 | DD-22-PZ-ST0022-20210325.mp4，1920×1080、30 FPS |
| 完成影格 | 360 |
| 等待人物／收集肩寬／鎖定 | 90／63／207 影格 |
| 鎖定目標 | OC-SORT ID 1，Selection_Generation 1 |
| 取得目標手部 | 207 影格 |
| 玩具模型有輸出 | 268 影格 |
| 手與玩具歷史框交集成立 | 204 影格 |
| unknown | 153 影格（等待及初始化） |
| 玩具推論裝置 | cuda:0 |

CSV 為 `00s - 05s: unknown`、`05s - 10s: playing`、`10s - 12s: playing`，最後一列標記部分窗口。
紀錄：`prefix_report.json`、`prefix_actions.csv`、`prefix_run.log`。

## 匯出版與畫面

```powershell
.\.venv\Scripts\python.exe -u video-save.py --headless --start-seconds 8 --max-frames 90 --output-csv logs/tracking_verification/export_actions.csv --export-video logs/tracking_verification/export_demo.mp4 --report logs/tracking_verification/export_report.json
```

完整輸出 90 個影格、1920×1080、30 FPS 的可解碼 MP4。初始化 63 個影格，後續 27 個影格鎖定 ID 1 並取得雙手，
其中 21 個影格有交集。CSV 保留原影片 `08s - 11s` 時間，標記部分窗口。
已檢視匯出最後一個影格：黃色框圈選小孩，雙手藍框回映至正確原圖位置；綠框為玩具輸出。
這個單幀目視確認不代表整段身分或行為準確率。

`export_preview.jpg` 為檢視用畫面，`export_report.json` 為統計，`export_demo.mp4` 為測試匯出。

另透過 `run_attention.cmd --max-frames 12 --start-seconds 8`，指定本測試資料夾的 CSV／report 輸出，
完成真實視窗建立、顯示及自動結束；紀錄為 `gui_report.json` 與 `gui_run.log`。

## 來源完整性及限制

已依 `TRACKING_SOURCE.json` 的 SHA256 確認 `medispipe_final` 來源檔案未修改；
複製的人物／姿態模型、肩寬選人與官方 OC-SORT 檔案亦與來源一致。
`tracking_pipeline.py` 僅新增可選姿態門檻開關。官方 OC-SORT 演算法及授權未修改。
舊 `attention.py`／`video-save.py` 的封存檔案雜湊仍與原基準一致。

本片段未觸發真實追蹤遺失／跨人重選，相關邊界由使用真正 OC-SORT 的合成整合測試與 runner 測試覆蓋。
未跑完整影片，沒有人工逐幀標註，不能將上述交集次數當作行為準確率。
