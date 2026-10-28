# attention 移植後執行檢查

> 本頁為改用 OC-SORT 前的歷史基準。當時兩支程式已保存在 `baseline_verification/source/`，
> `verify_baseline.py` 現在讀取該封存版本。現行程式與啟動方式請見 `README.md`；
> 以下舊版匯出失敗與 SHA256 紀錄不代表目前版本。

檢查日期：2026-10-27（Asia/Taipei）

## 結果

`attention.py` 已在本機專案環境通過 150 個影格（影片前 5 秒）的執行驗證。
`video-save.py` 仍有舊路徑，因此目前無法直接執行。
本次沒有修改兩支原始程式、模型、影片或偵測條件。

| 項目 | 結果 |
| --- | --- |
| 預設 `python attention.py` | 原先失敗：Anaconda base 缺少 `mediapipe`；亦未安裝 `ultralytics` |
| 專案 `.venv` | 已建立；`pip check` 通過 |
| 原始 Python 語法 | 兩支程式皆通過 |
| `best.pt` | 成功載入；類別為 `toy`、`basic`、`toy3D`、`basic3D` |
| YOLO GPU 推論 | 通過，`cuda:0`，NVIDIA GeForce RTX 5070 Ti |
| 影片讀取 | 通過；1920×1080、30 FPS、11,166 影格、372.2 秒 |
| MediaPipe Holistic | 成功初始化，150 個影格皆完成推論 |
| 原 `attention.main()` 流程 | 150 個影格皆完成 YOLO 與 MediaPipe 推論 |
| 手部偵測 | 左、右手各於 60 個影格有輸出 |
| 玩具偵測 | 59 個影格有模型輸出框 |
| 非零 IoU | 58 個影格觸發 |
| CSV 輸出 | `00s - 05s,playing` |
| OpenCV GUI | 真實視窗建立、顯示影格與關閉測試通過 |
| `video-save.py` 啟動 | 失敗，退出碼 1：舊權重路徑不存在 |

## 啟動原主程式

直接執行本資料夾的 `run_attention.cmd`，或在 PowerShell 執行：

```powershell
Set-Location 'C:\Users\User\Desktop\johnson\attention'
.\.venv\Scripts\python.exe .\attention.py
```

第一次偵測到人物後，程式會等待滑鼠框選受測者範圍，按 Enter 或空白鍵確認。
按 `q` 結束分析。原程式會將結果寫入本資料夾的 `action_log.csv`，並覆寫同名檔案。
影片使用相對路徑，因此手動啟動時需以上述資料夾為工作目錄；啟動檔已自動設定。

## 環境

本機原有 `C:\Users\User\anaconda3\envs\yolo` 提供 Python 3.11.13、
Ultralytics 8.3.161、PyTorch 2.9.1+cu128 與 torchvision 0.24.1+cu128。
專案 `.venv` 以 `--system-site-packages` 建立，沿用這些套件，另外在專案內安裝
MediaPipe 0.10.21、NumPy 1.26.4、OpenCV contrib 4.11.0.86 等相容依賴。
因此目前 `.venv` 仍依賴這台電腦的 conda `yolo` 環境，並非可以直接複製至其他電腦的獨立環境。
沒有在 conda `yolo` 或 Anaconda base 安裝、移除或升降版套件。

`requirements-baseline.txt` 記錄主要版本；`baseline-environment-lock.txt` 記錄完整可見套件版本。
下載套件、Ultralytics 設定及本次產物保留於本專案中。

此程式使用 `mp.solutions.holistic`；本次固定並實測 MediaPipe 0.10.21。
該版本提供 Windows CPython 3.11 wheel，見 [官方 PyPI 發布頁](https://pypi.org/project/mediapipe/0.10.21/)。

## 重做相同檢查

```powershell
Set-Location 'C:\Users\User\Desktop\johnson\attention'
.\.venv\Scripts\python.exe .\verify_baseline.py --frames 150 --gui-smoke
```

驗證工具載入並執行原 `attention.main()`，暫時限制影格數、自動提供全畫面 ROI，
並將結果導向 `baseline_verification`；未重寫偵測邏輯。
GUI 顯示另作短暫測試；未驗證使用者實際拖曳選框。

- `baseline_verification/verification_report.json`：版本、影片資訊、統計、CSV 內容與來源雜湊。
- `baseline_verification/action_log.csv`：本次 5 秒窗口的原程式輸出。
- `baseline_verification/video-save-startup.log`：匯出版啟動失敗的原始紀錄。

此結果確認短片段執行與資料輸出；尚未跑完整段 372.2 秒影片，也未驗證辨識準確率或是否選中正確小孩。

## 尚未移植的匯出版路徑

`video-save.py:10` 使用 `DD-04-PZ-ST0053-20201204.mov`，本資料夾沒有此影片。
第 11 行權重仍指向 `D:\johnson\J.H\runs\detect\train-7\weights\best.pt`；
實際啟動於載入此檔案時發生 `FileNotFoundError`。
本次只記錄問題，尚未替換其輸入影片或權重設定。

## 改進前的行為基準

- 觸發條件是 `IoU > 0`，任何非零交集都能觸發，未設定更高的交疊比率門檻。
- YOLO 使用逐幀 `model(frame)` 偵測，沒有物件追蹤 ID；保留約 0.5 秒的歷史玩具框。
- 每個 5 秒窗口只需累計至少 3 個觸發影格，即記為 `playing`，無須連續。
- MediaPipe 對整張畫面推論，ROI 僅事後按框中心篩選，沒有鎖定特定小孩的身分。
- YOLO 的四種類別輸出均作為玩具框使用，沒有依類別過濾。
- 結尾不足 5 秒的窗口不會寫入 CSV。

## 原始程式 SHA256

本次檢查前後一致：

```text
attention.py
8a9f061eefd6df1f5003b993fcfe5f99e55671a9d11744065ed721a2a6f59cf6

video-save.py
99184f6ccd68a8c771eb9c7cc3b90c777eb639079fa517b81f081c9a117d5a21
```
