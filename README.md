# LLM-Self-study

LLM의 사전 훈련, 미세조정, 검색 증강 생성과 추론 방법을 단계별로 정리하는 학습 저장소입니다.

## 학습 목차

1. [훈련 데이터 준비](./개념/1.%20훈련%20데이터%20준비/README.md)
   - 데이터 정제, 토큰화와 인코딩
   - 한국어 LLM 토크나이저
   - Dataset, DataLoader와 다음 토큰 예측
2. [미세조정](./개념/2.%20미세조정/README.md)
   - 전체 미세조정, LoRA와 QLoRA
   - SFT, 도메인 적응 추가 사전학습, 작업 특화 학습
   - 보안 로그 분석 데이터와 평가
   - 미세조정과 RAG의 관계
3. [데이터베이스 검색 기능 추가](./개념/3.%20데이터베이스%20검색%20기능%20추가/README.md)
4. [내부적 질의를 통한 더 좋은 결론 도출](./개념/4.%20내부적%20질의를%20통한%20더%20좋은%20결론%20도출/README.md)
5. [모델 정의](./개념/모델%20정의/README.md)
   - Transformer, Masked Multi-Head Attention, Feed-Forward, Add & Norm
   - 활성화 함수, Temperature와 주요 모델 상수
   - PyTorch 주요 모듈과 Decoder-only 모델 예시

## 저장소 구조

```text
LLM-Self-study/
├─ README.md
└─ 개념/
   ├─ 1. 훈련 데이터 준비/
   │  └─ README.md
   ├─ 2. 미세조정/
   │  └─ README.md
   ├─ 3. 데이터베이스 검색 기능 추가/
   │  └─ README.md
   ├─ 4. 내부적 질의를 통한 더 좋은 결론 도출/
   │  └─ README.md
   └─ 모델 정의/
      └─ README.md
```

각 목차의 링크는 현재 브랜치를 기준으로 하는 상대 경로를 사용합니다. 문서가 아직 기본 브랜치에 병합되지 않은 경우에는 GitHub에서 `사전-훈련` 브랜치를 선택해야 확인할 수 있습니다.
