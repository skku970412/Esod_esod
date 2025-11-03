# 환경 설정 보고서

## 가상 환경
- 위치: `.venv` (Python 3.10.12). `source .venv/bin/activate`로 활성화합니다.
- 핵심 패키지: `torch==2.2.1+cpu`, `torchvision==0.17.1+cpu`, `opencv-contrib-python-headless==4.7.0.72`, `pytest==8.4.2`를 포함해 `esod/requirements.txt`의 의존성을 설치했습니다.

## 의존성 관리
- 가상 환경 안에서 `pip install -r esod/requirements.txt`로 기본 의존성을 설치했습니다.
- `cv2.dnn.DictValue` 임포트 오류를 피하기 위해 PyTorch CPU 휠을 고정하고 기본 OpenCV 패키지를 contrib headless 버전으로 교체했습니다.
- `pip freeze > esod/requirements.lock`으로 전체 환경을 고정했습니다.

## 테스트
- 중요한 모듈 임포트를 검증하는 간단한 스모크 테스트를 `esod/tests/test_imports.py`에 추가했습니다.
- 가상 환경에서 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_imports.py -q`를 실행했고 약 6.6초 만에 `2 passed`를 확인했습니다. 해당 환경 변수는 서드파티 pytest 플러그인이 컬렉션을 방해하지 않도록 합니다.

## 참고 및 다음 단계
- GPViT 어댑터에 필요한 `mmdet` 등 선택 패키지는 설치하지 않았으니 필요 시 추가하세요.
- 이후 스크립트나 테스트를 실행할 때는 항상 `source .venv/bin/activate`로 가상 환경을 활성화하세요.
- 새로운 패키지를 설치한 뒤에는 `esod/requirements.lock`을 갱신해 재현 가능성을 유지하세요.
