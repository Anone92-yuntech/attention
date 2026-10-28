# Model assets

Download the three assets from the public [v1.0.0 release](https://github.com/Anone92-yuntech/attention/releases/tag/v1.0.0),
or run `python fetch_models.py` from the project folder. The script places each file at the path below and verifies its size and SHA256.
Model binaries are excluded from Git; they are distributed through GitHub Releases.

| Release asset | Local path | Bytes | SHA256 |
| --- | --- | ---: | --- |
| best.pt | best.pt | 40516645 | f5fce7480e8fb4aebd800eeaae24299979cd7aa6df49c8085b0a67ee6ecf906f |
| yolox_nano.onnx | models/yolox_nano.onnx | 3659407 | c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d |
| pose_landmarker_lite.task | models/pose_landmarker_lite.task | 5777746 | 59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a |

`best.pt` is the migrated project's trained four-class toy detector. YOLOX-Nano is the official COCO person detector
from the YOLOX 0.1.1rc0 release; Pose Landmarker Lite comes from the MediaPipe model asset.
The files are unchanged copies of the original project assets. See `TRACKING_SOURCE.json` and `person_detector.py` for source metadata.

Input videos are not distributed. Supply your own video with `--input`.
