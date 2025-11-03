**ESOD: Efficient Small Object Detection on High-Resolution Images**

이 저장소는 논문 [ESOD: Efficient Small Object Detection on High-Resolution Images](https://arxiv.org/abs/2407.16424)의 구현체입니다.  
아래 내용은 Arch 기반 컨테이너(Python 3.10.12, CUDA 12.1)에서 재현한 절차를 정리했습니다.

> **표기 규칙**  
> - 별도 언급이 없으면 모든 명령은 `/home/work/llama_young/esod_sizak/esod` 디렉터리에서 실행합니다.  
> - 상위 경로에서 작업할 때는 명령 블록에 전체 경로를 함께 표기했습니다.

---

## 0. 3분 요약 (venv 생성부터 10장 추론까지)

```bash
# 0) 프로젝트 루트로 이동
cd /home/work/llama_young/esod_sizak

# 1) 가상환경 생성 및 활성화
python3 -m venv .venv --without-pip
.venv/bin/python get-pip.py           # 최초 1회만
source .venv/bin/activate

# 2) 필수 패키지 설치
pip install --upgrade pip
pip install -r esod/requirements.txt
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# 3) ESOD 디렉터리로 이동해 VisDrone 링크 확인
cd esod
ln -sfn /home/work/llama_young/VisDrone VisDrone

# 4) (선택) 전처리 수행 - SAM 없이 빠르게 확인하려면 생략 가능
python scripts/data_prepare.py --dataset VisDrone

# 5) 검증용 이미지 10장 목록 만들기
head -n 10 VisDrone/split/val.txt > VisDrone/split/val_10.txt

# 6) 10장만 추론 실행
python detect.py \
  --weights ../esod_yolov8m.pt \
  --source VisDrone/split/val_10.txt \
  --img-size 1536 --device 0 \
  --project runs/detect --name visdrone_val10 --exist-ok
```

`runs/detect/visdrone_val10/`에 결과 이미지가 생성되면 환경이 정상 동작 중입니다.

---

## 1. 환경 구성

```bash
cd /home/work/llama_young/esod_sizak
python3 -m venv .venv --without-pip
.venv/bin/python get-pip.py        # 이미 실행했다면 생략 가능
source .venv/bin/activate

pip install --upgrade pip
pip install -r esod/requirements.txt
# GPU 사용 시
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# 논문과 동일한 GT-heatmap 품질이 필요하다면 (선택)
pip install -e esod/third_party/segment-anything
curl -L -o esod/weights/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

- `pip freeze > esod/requirements.lock`로 현재 환경을 저장했습니다.
- SAM 설치는 필수가 아니며, GT 마스크 품질을 높이고 싶을 때만 사용합니다.

---

## 2. 데이터 준비

### 2.1 VisDrone (기본 실험)
1. `/home/work/llama_young`에서 압축 해제  
   ```
   unzip VisDrone2019-DET-*.zip -d VisDrone
   ```
2. ESOD 루트에서 연결  
   ```bash
   cd esod
   ln -sfn /home/work/llama_young/VisDrone VisDrone
   ```
3. 전처리  
   - SAM 사용: `python scripts/data_prepare.py --dataset VisDrone`  
     (이미지당 수초~수십초 소요, GPU 24GB 이상 권장)
   - SAM 미사용: 스크립트에서 `predictor` 분기를 비활성화하거나 기존 라벨을 유지

### 2.2 UAVDT / TinyPerson
```bash
python scripts/data_prepare.py --dataset UAVDT
python scripts/data_prepare.py --dataset TinyPerson
```

---

## 3. 사전학습 가중치

| 모델 | 경로 |
| --- | --- |
| ESOD-YOLOv5m (VisDrone) | `/home/work/llama_young/esod_sizak/esod_yolov5m.pt` |
| ESOD-YOLOv8m (VisDrone) | `/home/work/llama_young/esod_sizak/esod_yolov8m.pt` |
| YOLOv5m 기본 | `weights/yolov5m.pt` |

필요 시 `weights/` 아래로 이동해 사용하세요.

---

## 4. 빠른 검증 (샘플 10장)

데이터 준비 직후 파이프라인 동작 여부를 확인할 때 사용합니다.

```bash
cd esod
head -n 10 VisDrone/split/val.txt > VisDrone/split/val_10.txt
cat > data/visdrone_val10.yaml <<'EOF'
train: ./VisDrone/split/train.txt
val: ./VisDrone/split/val_10.txt
test: ./VisDrone/split/val_10.txt
nc: 80
names: ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
        'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
        'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
        'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
        'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
        'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
        'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard',
        'cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase',
        'scissors','teddy bear','hair drier','toothbrush']
EOF

python test.py \
  --weights ../esod_yolov5m.pt \
  --data data/visdrone_val10.yaml \
  --img-size 1536 --batch-size 1 --device 0 \
  --task val --save-txt \
  --project runs/test --name visdrone_val10_esodv5m --exist-ok
```

출력은 `runs/test/visdrone_val10_esodv5m`에 저장됩니다.  
`../esod_yolov8m.pt`을 사용하면 ESOD-YOLOv8m 결과를 얻을 수 있습니다.

---

## 5. 전체 검증 및 속도 측정

```bash
# 전체 val 평가 (VisDrone)
python test.py \
  --weights ../esod_yolov8m.pt \
  --data data/visdrone.yaml \
  --img-size 1536 --batch-size 8 --device 0 \
  --task val --save-txt \
  --project runs/test --name visdrone_full_esodv8m

# FLOPs / FPS 측정
python test.py \
  --weights ../esod_yolov8m.pt \
  --data data/visdrone.yaml \
  --img-size 1536 --batch-size 1 --device 0 \
  --task measure
```

논문 수치를 재현하려면 V100 GPU ×2, 동일한 입력 크기, SAM 기반 GT 마스크를 사용하는 것이 이상적입니다.

### 5.1 사전학습 ESOD 가중치 벤치마크 모음

다음 명령으로 ESOD 모델의 주요 설정을 빠르게 비교할 수 있습니다. 각 실행 시 결과는 `runs/test/<name>/metrics.log`에 저장됩니다.

```bash
cd /home/work/llama_young/esod_sizak/esod
source ../.venv/bin/activate

# ESOD YOLOv5m @1536
python test.py --weights ../esod_yolov5m.pt --data data/visdrone.yaml \
  --img-size 1536 --batch-size 8 --device 0 \
  --task val --save-txt --project runs/test \
  --name visdrone_full_esod_yolov5m --exist-ok \
  | tee runs/test/visdrone_full_esod_yolov5m/metrics.log

# ESOD YOLOv8m @1536
python test.py --weights ../esod_yolov8m.pt --data data/visdrone.yaml \
  --img-size 1536 --batch-size 8 --device 0 \
  --task val --save-txt --project runs/test \
  --name visdrone_full_esod_yolov8m --exist-ok \
  | tee runs/test/visdrone_full_esod_yolov8m/metrics.log

# ESOD YOLOv5m @1920 (해상도 1.25×)
python test.py --weights ../esod_yolov5m.pt --data data/visdrone.yaml \
  --img-size 1920 --batch-size 4 --device 0 \
  --task val --save-txt --project runs/test \
  --name visdrone_full_esod_yolov5m_1920 --exist-ok \
  | tee runs/test/visdrone_full_esod_yolov5m_1920/metrics.log

# ESOD YOLOv8m @1920 (해상도 1.25×)
python test.py --weights ../esod_yolov8m.pt --data data/visdrone.yaml \
  --img-size 1920 --batch-size 4 --device 0 \
  --task val --save-txt --project runs/test \
  --name visdrone_full_esod_yolov8m_1920 --exist-ok \
  | tee runs/test/visdrone_full_esod_yolov8m_1920/metrics.log
```

---

## 6. 추론 스크립트

```bash
python detect.py \
  --weights ../esod_yolov8m.pt \
  --source VisDrone/VisDrone2019-DET-val/images \
  --img-size 1536 --device 0 \
  --view-cluster --line-thickness 1
```

출력은 `runs/detect/` 아래에 저장되며, `--view-cluster` 옵션을 켜면 슬라이스 패치, 예측/GT 히트맵을 동시에 확인할 수 있습니다.

---

## 7. SAM 관련 메모

- 논문 학습 재현을 위해 권장되지만, 순수 추론(inference)에는 필요하지 않습니다.
- 이미지 해상도가 높아 SAM 추론 시간이 길고, GPU 메모리 요구량도 큰 편입니다.
- OOM이 발생하면 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` 등으로 메모리 조각화를 완화하세요.

---

## 8. FAQ

- **`cv2` 모듈 오류**: `opencv-python>=4.1.1`이 필요합니다. venv 기준 `pip install opencv-python-headless==4.8.0.76`으로 교체하면 해결됩니다.
- **`ConfusionMatrix` IndexError**: COCO 80 클래스 가중치에 10 클래스 YAML을 사용한 경우입니다. `nc`와 `names` 배열을 80개로 맞추면 됩니다.
- **추론만 하고 싶은데 SAM이 필요하나요?**: 아닙니다. 기존 가중치로 바로 `test.py`/`detect.py`를 실행하면 됩니다.

---

## 9. 인용

```
@article{liu2025esod,
  title={ESOD: Efficient Small Object Detection on High-Resolution Images},
  author={Liu, Kai and Fu, Zhihang and Jin, Sheng and Chen, Ze and Zhou, Fan and Jiang, Rongxin and Chen, Yaowu and Ye, Jieping},
  journal={IEEE Transactions on Image Processing},
  volume={34},
  pages={183--195},
  year={2025}
}
```

도움이 되었다면 별표와 인용 부탁드립니다!


