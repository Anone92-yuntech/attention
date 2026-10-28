# attention：雙肩選人與 OC-SORT 追蹤

已將原本手動框選 ROI 的方式，改成 `medispipe_final` 專案的肩寬選人、真正 OC-SORT 追蹤及目標專用姿態流程。
`best.pt` 繼續偵測玩具；雙手只從當幀有效目標的人物裁切畫面取得。

## 啟動

首次使用請先完成下方環境安裝。GitHub 不提供兒童原始影片、影格或分析輸出，啟動時請用 `--input` 指定自己的影片。
程式會自動收集肩寬並選人，不再要求手動框選。
黃色框標示目標 ID，藍色框是目標雙手，綠色框是玩具。

| 按鍵 | 功能 |
| --- | --- |
| 空白鍵 | 暫停／繼續；暫停不推進影片時間 |
| R | 清除選人與手部歷史，重新收集肩寬；保留 OC-SORT 與影片時間 |
| Q／Esc／關閉視窗 | 結束，關閉資源並保留最後一個部分 CSV 窗口 |

在專案資料夾開啟 PowerShell，指定本機影片：

```powershell
.\run_attention.cmd --input 'D:\影片\測試.mp4'
```

預設結果為本資料夾的 `action_log.csv`。輸出路徑可用 `--output-csv` 指定；重複使用同一路徑會覆寫該輸出。
相對路徑一律從本專案資料夾解析，無須先手動切換工作目錄。

## 選人及追蹤規則

1. 使用與來源專案相同的 YOLOX-Nano 人物模型，將人物框交給官方 OC-SORT；每個來源影格都更新，包括空偵測影格。
2. 約每 3 個影格逐人裁切，使用 MediaPipe Tasks Pose 取得雙肩；肩膀 visibility 與 presence 均須至少 0.7，並檢查幾何位置與其他人物的歧義。
3. 累積約 2 秒、至少 10 份有效樣本，以原影片像素的肩寬中位數比較。僅一人合格時選該人；多人合格時選最小者，與第二小者差距不足 10% 就繼續觀察。
4. 鎖定後沿用 OC-SORT ID，不因後續肩寬排序改變而換人。只允許當幀有真正偵測配對的目標進入手部分析，預測框不作行為證據。
5. Holistic 僅分析該目標裁切；雙肩還須與 Tasks 目標肩位相符，雙手框再回映原圖。更換 ID、重新選人代次、遺失或姿態不可信時，關閉舊手部模型並清除玩具緩衝。

沿用來源的恢復預設：漏掉目標當幀有效配對即重新收集肩寬；ID 仍存在但姿態持續不足時，約每 2 秒重建目標 VIDEO pose，連續不足達 10 秒重新選人。
重新選人會增加 `Selection_Generation`，OC-SORT 不重置。

如需短暫漏檢緩衝或保留原 ID 等待，可使用：

```powershell
.\run_attention.cmd --auto-reset-seconds 0.5
.\run_attention.cmd --no-reselect-on-loss
```

肩寬是選人規則，並非年齡辨識；重選後也可能選到其他人。多人遮擋與 ID 交換仍需人工檢視。

## 手與玩具判定及 CSV

保留原本的交集條件與時間規則：雙手框向外增加 5%，玩具框保留約 0.5 秒；預設 `IoU > 0` 觸發。
每個 5 秒窗口累計至少 3 個觸發影格，記為 `playing`，不要求連續。
仍使用 `best.pt` 全部四種類別（`toy`、`basic`、`toy3D`、`basic3D`）。玩具在完整原圖偵測，不再受舊手動 ROI 的中心點限制。

`Time_Window` 與 `Status` 仍是 CSV 前兩欄，另外記錄目標 ID、選人代次、有效／未知／觸發影格、混合身分與部分窗口。
尚未選定目標或目標不可信時為 `unknown`；不同 ID 或代次不能合併觸發次數，跨身分窗口整列為 `unknown`。
有效樣本不足以確認未發生行為時，也不直接寫成 `none`。
末尾不足 5 秒會輸出並標記 `Partial_Window=True`；切換身分前捨棄的樣本列於 `Discarded_Frames`。

可調整交集門檻，例如：

```powershell
.\run_attention.cmd --iou-threshold 0.1
```

## 影片匯出

`video-save.py` 已改用同一套分析流程與目前資料夾的影片／權重，不再使用舊電腦的 D 槽路徑。
以 `--input` 指定自己的影片，預設輸出 `output_demo.mp4` 及 `action_log.csv`。

```powershell
.\run_video_save.cmd --input 'D:\影片\測試.mp4' --export-video '展示結果.mp4'
```

## 環境與來源

以下為 Windows 64 位元、Python 3.11 的乾淨安裝流程。從 GitHub clone 或下載 ZIP 後，
進入包含 `attention.py` 的專案資料夾並開啟 PowerShell。先安裝 Python 3.11，再建立獨立環境；不需複製原電腦的 conda 或 `.venv`。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

先安裝 PyTorch。一般電腦可使用 CPU 版本：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
```

若要使用 NVIDIA GPU，且驅動支援 CUDA 12.8，將上面的 PyTorch 安裝指令改成：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

這兩組版本與指令見 [PyTorch 官方歷史版本頁](https://pytorch.org/get-started/previous-versions/)。
選擇其中一種後，再安裝本專案套件並檢查依賴：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

主要版本為 MediaPipe 0.10.21、Ultralytics 8.3.161 與 FilterPy 1.4.5。
人物 YOLOX 使用 OpenCV CPU；玩具 YOLO 使用可用的 PyTorch CPU 或 CUDA 裝置。
原電腦曾使用 Python 3.11.13、PyTorch 2.9.1+cu128 及繼承 conda 的 `.venv`，屬於先前驗證環境。
`baseline-environment-lock.txt`、`tracking-environment-lock.txt` 僅為該電腦的版本快照，包含本機 `file:///C:/...` 來源，不能拿來執行 `pip install -r`；乾淨安裝請使用上列流程與 `requirements.txt`。

推論所需的三個模型放在 [v1.0.0 Release](https://github.com/Anone92-yuntech/attention/releases/tag/v1.0.0)，不放入 Git 歷史。
安裝環境後執行以下指令，會下載並比對檔案大小及 SHA256；本機已有正確模型時會跳過：

```powershell
.\.venv\Scripts\python.exe .\fetch_models.py
.\.venv\Scripts\python.exe .\fetch_models.py --verify-only
.\run_attention.cmd --input 'D:\影片\測試.mp4'
```

三個模型的路徑與雜湊見 `models/README.md`。輸入影片須自行準備，範例路徑請改成自己的實際檔案。
GitHub 不提供兒童原始影片、畫面截圖、CSV 或執行紀錄；這些資料均留在原電腦。

選人、人物偵測、OC-SORT 包裝與姿態流程從 `medispipe_final` 建立本專案快照；所需兩份人物／姿態模型保存在本專案 `models/`。
來源專案未修改。`TRACKING_SOURCE.json` 記錄來源 SHA256，`vendor/ocsort` 保留官方演算法、MIT 授權與固定 commit 的來源資訊。
姿態流程新增可選開關，使本專案不必滿足與玩具行為無關的趴桌關鍵點條件；肩寬選人與 OC-SORT 演算法維持來源版本。

舊版程式保存在 `baseline_verification/source/`；`BASELINE_CHECK.md` 與 `verify_baseline.py` 對應改進前的封存基準。
重跑封存基準另需在本機放入原始同名影片；該影片不隨 GitHub 專案提供。
新入口為 `attention.py`／`video-save.py`，共用 `attention_app.py`、`attention_engine.py` 與 `attention_log.py`。

## 驗證

追蹤整合的 149 項測試通過；原影片前 360 個影格（12 秒）自動鎖定 ID 1，完成目標手部與玩具分析及 CSV 輸出。
另驗證 90 個影格的影片匯出，以及啟動檔的真實 GUI 顯示。完整數據見 `VERIFICATION.md`。
這些為功能與片段執行驗證，尚未驗證完整影片的追蹤正確率或行為辨識準確率。

GitHub 發佈另增加 11 項模型下載及完整性測試，現版本共 160 項測試通過。模型下載測試使用模擬網路，不會上傳或下載私人影片。
