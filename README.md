# ESOD Quickstart (Root Workspace)

이 문서는 `/home/work/llama_young/esod_sizak` 루트 디렉터리에서 ESOD 관련 스크립트를 실행할 때 필요한 최소한의 정보만 정리했습니다. 세부 실험 가이드와 코드 설명은 하위 디렉터리 `esod/README.md`를 참고하세요.

## 1. 기본 환경 준비

```bash
cd /home/work/llama_young/esod_sizak
python3 -m venv .venv --without-pip
./.venv/bin/python get-pip.py
source .venv/bin/activate
pip install --upgrade pip
pip install -r esod/requirements.txt
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

## 2. VisDrone 데이터 연결 및 전처리 (선택)

```bash
cd /home/work/llama_young/esod_sizak/esod
ln -sfn /home/work/llama_young/VisDrone VisDrone

# SAM 기반 GT가 필요할 때 (val 세트 기준)
../prepare_val_with_sam.sh
```

train+val 전체를 새로 만들고 싶으면 압축을 새로 풀어 동일 스크립트를 적용하세요.

## 3. 루트에서 바로 실행 가능한 스크립트

| 스크립트 | 설명 | 결과/로그 |
| --- | --- | --- |
| `quick_run.sh` | venv + 의존성 설치 → VisDrone 링크 → (선택) SAM 전처리 → 10장 추론 | `esod/runs/detect/visdrone_val10/` |
| `run_esod_yolov5_eval.sh` | ESOD YOLOv5m @1536 전체 val 평가 | `esod/runs/test/visdrone_full_esod_yolov5m/metrics.log` |
| `run_metrics_full.sh` | ESOD YOLOv8m @1536 평가 | `esod/runs/test/visdrone_full_esodv8m/` |
| `run_metrics_full_sam.sh` | SAM 전처리 데이터를 이용한 ESOD YOLOv8m 평가 | `esod/runs/test/visdrone_full_esodv8m_sam/` |
| `run_metrics_full_yolov5.sh` | COCO YOLOv5 baseline 평가 (VisDrone fine-tuning 필요) | `esod/runs/test/visdrone_full_yolov5m_base/` |
| `run_metrics_full_yolov8_baseline.sh` | Ultralytics YOLOv8 baseline 평가 | `esod/runs/test/visdrone_full_yolov8_baseline/` |
| `run_metrics_full_yolov8_ultra.sh` | Ultralytics CLI 기반 YOLOv8 baseline 평가 | `esod/runs/ultra/visdrone_full_yolov8_baseline/` |
| `run_all_benchmarks.sh` | ESOD YOLOv5/YOLOv8 (1536 & 1920) 일괄 실행 | `esod/runs/benchmark/` |
| `compare_yolov5m.sh` | ESOD YOLOv5m vs COCO YOLOv5m 비교 (클래스 불일치 주의) | `esod/runs/test/visdrone_full_yolov5m_*` |

각 스크립트는 내부에서 venv 활성화를 호출하거나 가정하고 있으며, `tee`를 이용해 로그를 `metrics.log`로 저장합니다.

## 4. 수동 실행 예시

```bash
cd /home/work/llama_young/esod_sizak/esod
source ../.venv/bin/activate

# ESOD YOLOv5m @1536
python test.py --weights ../esod_yolov5m.pt --data data/visdrone.yaml \
  --img-size 1536 --batch-size 8 --device 0 \
  --task val --save-txt --project runs/test \
  --name visdrone_full_esod_yolov5m --exist-ok \
  | tee runs/test/visdrone_full_esod_yolov5m/metrics.log

# ESOD YOLOv8m @1920 (1.25×)
python test.py --weights ../esod_yolov8m.pt --data data/visdrone.yaml \
  --img-size 1920 --batch-size 4 --device 0 \
  --task val --save-txt --project runs/test \
  --name visdrone_full_esod_yolov8m_1920 --exist-ok \
  | tee runs/test/visdrone_full_esod_yolov8m_1920/metrics.log
```

## 5. 참고 사항

- COCO 사전학습 가중치(`yolov5m.pt`, `yolov8m.pt`)는 VisDrone 10클래스와 라벨 구조가 다릅니다. 논문 수준 비교를 위해서는 VisDrone 데이터로 별도 fine-tuning이 필요합니다.
- SAM 기반 전처리는 24 GB GPU 기준 val 548장을 처리하는 데 약 3분 정도 걸립니다. train까지 포함하면 그 몇 배의 시간이 필요합니다.

---

세부 파이프라인과 추가 실험 명령은 `esod/README.md`를 참고하세요.
