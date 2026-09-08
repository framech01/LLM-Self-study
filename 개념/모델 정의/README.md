# 3. 모델 정의

LLM 모델 정의는 입력 토큰 ID를 벡터로 변환하고, Transformer 블록을 반복해 문맥을 처리한 뒤, 다음 토큰에 대한 확률 분포를 출력하도록 신경망의 구조와 하이퍼파라미터를 정하는 과정이다.

## 3.1 전체 처리 흐름

일반적인 자기회귀 언어 모델의 흐름은 다음과 같다.

```text
토큰 ID
  ↓
토큰 임베딩 + 위치 정보
  ↓
Transformer 블록 × N
  ├─ Masked Multi-Head Self-Attention
  ├─ 잔차 연결 + 정규화
  ├─ Feed-Forward Network
  └─ 잔차 연결 + 정규화
  ↓
출력 정규화
  ↓
Linear(vocab_size)
  ↓
로짓(logits)
  ↓
Softmax 및 샘플링
  ↓
다음 토큰
```

첨부된 그림은 「Attention Is All You Need」 논문의 원본 **인코더–디코더 Transformer** 구조다.

- 왼쪽 인코더는 입력 전체를 양방향으로 참조한다.
- 오른쪽 디코더의 첫 attention은 미래 토큰을 볼 수 없도록 마스킹한다.
- 디코더의 두 번째 attention은 인코더의 출력에 접근하는 cross-attention이다.
- 오늘날 GPT 계열의 자기회귀 LLM은 일반적으로 오른쪽과 유사한 **decoder-only 구조**를 사용하지만, 인코더 출력이 없으므로 cross-attention은 보통 제외한다.
- BERT 계열은 주로 encoder-only, 번역용 T5 계열은 encoder-decoder 구조를 사용한다.

## 3.2 임베딩과 위치 정보

### 토큰 임베딩

`nn.Embedding`은 정수 토큰 ID를 `d_model` 차원의 밀집 벡터로 변환한다.

```text
token_id: 315 → [0.12, -0.44, ..., 0.08]
```

임베딩 행렬의 크기는 일반적으로 `vocab_size × d_model`이다. `vocab_size`가 커지면 더 많은 토큰을 직접 표현할 수 있지만 임베딩과 출력 계층의 파라미터 수도 증가한다.

### 위치 정보

Self-Attention 자체는 토큰의 순서를 알지 못하므로 위치 정보를 추가해야 한다.

- 원본 Transformer: 사인·코사인 기반 positional encoding
- 학습형 positional embedding: 위치마다 학습 가능한 벡터 사용
- RoPE(Rotary Position Embedding): 현대 LLM에서 널리 사용되는 회전 기반 상대 위치 표현

## 3.3 Multi-Head Self-Attention

Self-Attention은 같은 시퀀스에서 만든 Query, Key, Value를 이용해 각 토큰이 다른 토큰을 얼마나 참고할지 계산한다.

```text
Q = XW_Q
K = XW_K
V = XW_V

Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

- **Query(Q)**: 현재 토큰이 찾고자 하는 정보
- **Key(K)**: 각 토큰이 어떤 정보를 가졌는지 나타내는 색인
- **Value(V)**: attention 가중치에 따라 실제로 조합할 정보
- `1 / √d_k`: 내적 값이 지나치게 커져 Softmax가 포화되는 것을 완화하는 스케일링

### Multi-Head인 이유

하나의 attention만 사용하는 대신 여러 head가 서로 다른 투영 공간에서 관계를 병렬로 학습한다. 어떤 head는 가까운 문법 관계를, 다른 head는 장거리 문맥이나 의미 관계를 학습할 수 있다.

```text
head_i = Attention(XW_Qᵢ, XW_Kᵢ, XW_Vᵢ)
MultiHead(X) = Concat(head_1, ..., head_h)W_O
```

일반적으로 `d_model`은 `n_heads`로 나누어떨어져야 하며, 각 head의 차원은 `head_dim = d_model / n_heads`가 된다.

## 3.4 Masked Multi-Head Attention

자기회귀 LLM은 현재 위치에서 미래 정답 토큰을 미리 보면 안 된다. 이를 막기 위해 attention score에 **causal mask**를 적용한다.

```text
        볼 수 있는 위치
토큰 1  [O, X, X, X]
토큰 2  [O, O, X, X]
토큰 3  [O, O, O, X]
토큰 4  [O, O, O, O]
```

미래 위치의 score에 `-∞`에 가까운 값을 더하면 Softmax 이후 확률이 0이 된다.

```text
MaskedAttention(Q, K, V)
  = softmax(QKᵀ / √d_k + causal_mask) V
```

이 마스크 덕분에 훈련 중에는 전체 시퀀스를 병렬로 계산하면서도 각 위치가 이전 토큰만 보고 다음 토큰을 예측하도록 만들 수 있다. 패딩 토큰을 무시하기 위한 padding mask와 미래 토큰을 가리는 causal mask는 목적이 다르다.

## 3.5 Feed-Forward Network

Feed-Forward Network(FFN)는 attention으로 모은 정보를 각 토큰 위치에서 독립적으로 변환하는 MLP다. 모든 위치에 같은 가중치를 적용한다.

원본 Transformer의 기본 형태는 다음과 같다.

```text
FFN(x) = Linear₂(Activation(Linear₁(x)))
```

- 첫 번째 Linear: `d_model`에서 더 큰 `d_ff` 차원으로 확장
- 활성화 함수: 비선형성을 추가
- 두 번째 Linear: 다시 `d_model` 차원으로 축소
- 선택적 Dropout: 학습 중 일부 활성화를 무작위로 제거

현대 LLM에서는 GELU 기반 MLP 외에도 SiLU와 gating을 결합한 SwiGLU 같은 구조를 많이 사용한다.

## 3.6 활성화 함수

활성화 함수는 선형 계층만으로 표현할 수 없는 비선형 관계를 학습하게 한다.

### ReLU

```text
ReLU(x) = max(0, x)
```

- 계산이 단순하고 빠르다.
- 음수 영역의 출력과 기울기가 0이어서 뉴런이 더 이상 학습하지 않는 dead ReLU 문제가 생길 수 있다.

### GELU

입력을 0 또는 그대로 통과시키는 대신 값의 크기에 따라 부드럽게 조절한다. BERT와 초기 GPT 계열 Transformer에서 널리 사용됐다.

### SiLU 또는 Swish

```text
SiLU(x) = x · sigmoid(x)
```

부드러운 활성화 함수이며, SwiGLU와 함께 현대 LLM의 FFN에서 자주 사용된다.

## 3.7 Add & Norm

그림의 `Add & Norm`은 **잔차 연결(Residual Connection)**과 **정규화(Normalization)**를 의미한다.

### Add: 잔차 연결

하위 계층의 입력 `x`를 계층의 출력에 더한다.

```text
y = x + Sublayer(x)
```

잔차 연결은 깊은 신경망에서 정보와 기울기가 여러 계층을 안정적으로 통과하도록 돕는다. 더하려는 두 텐서의 마지막 차원은 같아야 한다.

### Norm: 정규화

`LayerNorm`이나 `RMSNorm`은 주로 각 토큰의 **활성값(hidden state)**을 특성 차원 기준으로 정규화한다. 일반적인 `Add & Norm`의 Norm을 단순히 “가중치 값을 정규화한다”라고 설명하는 것은 정확하지 않다. 학습 가능한 scale이나 bias 파라미터는 포함될 수 있지만, 핵심 대상은 계층을 통과하는 활성값이다.

- **LayerNorm**: 평균과 분산을 이용해 정규화
- **RMSNorm**: 평균을 빼지 않고 RMS 크기를 이용해 정규화

원본 Transformer 그림은 sublayer 뒤에 정규화하는 Post-Norm 구조다.

```text
Post-Norm: y = Norm(x + Sublayer(x))
```

현대 LLM에서는 깊은 모델의 학습 안정성을 위해 정규화를 먼저 수행하는 Pre-Norm 구조가 흔하다.

```text
Pre-Norm: y = x + Sublayer(Norm(x))
```

## 3.8 출력 계층과 Softmax

마지막 hidden state는 선형 계층을 통과해 `vocab_size`개의 로짓으로 변환된다.

```text
hidden state: [batch, sequence, d_model]
logits:       [batch, sequence, vocab_size]
```

로짓은 정규화되지 않은 점수다. 훈련 시에는 일반적으로 `CrossEntropyLoss`에 로짓을 직접 전달하며, 손실 함수 내부에서 Log-Softmax에 해당하는 계산이 수행된다. 따라서 훈련 코드에서 Softmax를 먼저 적용하면 안 된다.

추론 시에는 마지막 위치의 로짓을 Softmax로 확률 분포로 변환한 다음 greedy decoding 또는 확률적 샘플링으로 다음 토큰을 선택한다.

## 3.9 LLM Temperature

Temperature는 **추론 시** 다음 토큰 확률 분포의 뾰족한 정도를 조절하는 값이다.

```text
probabilities = softmax(logits / temperature)
```

- **작은 값(`0 < T < 1`)**: 로짓 차이가 확대되어 확률이 높은 토큰에 더 집중한다. 결과가 일관적이고 보수적으로 변한다.
- **큰 값(`T > 1`)**: 로짓 차이가 축소되어 분포가 평평해진다. 다양한 토큰이 선택될 가능성이 커진다.
- **`T = 1`**: 원래 로짓 분포를 그대로 사용한다.
- **`T = 0`**: 수식에서 0으로 나눌 수 없으므로 실제 구현에서는 보통 temperature 계산 대신 가장 큰 로짓을 고르는 greedy decoding으로 처리한다.

Temperature가 모델의 지식을 늘리거나 정확성을 보장하는 것은 아니다. 또한 greedy decoding을 사용하면 확률 샘플링을 하지 않기 때문에 temperature의 영향이 없다. 실제 생성에서는 `top-k`, `top-p` 같은 샘플링 제한과 함께 사용한다.

## 3.10 모델 정의에 사용하는 주요 상수

`VOBAB_Size`가 아니라 **`vocab_size`**가 일반적인 표기다.

| 설정 | 의미 |
|---|---|
| `vocab_size` | 토크나이저 어휘 사전에 포함된 토큰 수 |
| `context_length` 또는 `max_seq_len` | 모델이 한 번에 처리할 수 있는 최대 토큰 길이 |
| `d_model` 또는 `hidden_size` | 각 토큰 hidden state의 차원 |
| `n_heads` | Multi-Head Attention의 head 수 |
| `n_layers` | 반복할 Transformer 블록 수 |
| `d_ff` 또는 `intermediate_size` | FFN 내부 확장 차원 |
| `dropout` | 학습 중 무작위로 제거할 활성값의 비율 |
| `norm_eps` | 정규화 계산에서 0으로 나누는 것을 방지할 작은 값 |
| `qkv_bias` | Q, K, V 선형 투영에 bias를 사용할지 여부 |

모델 크기는 이 설정들의 조합으로 결정된다. 특히 `d_model`, `n_layers`, `d_ff`, `vocab_size`를 키우면 파라미터 수와 메모리·연산 비용도 증가한다.

## 3.11 PyTorch란?

PyTorch는 텐서 연산, GPU 가속, 자동 미분과 신경망 구성 기능을 제공하는 오픈소스 딥러닝 프레임워크다.

- **텐서와 GPU 가속**: 다차원 배열을 CPU, CUDA GPU 등의 장치에서 계산한다.
- **동적 계산 그래프**: 기본 eager 실행에서는 Python 제어 흐름에 따라 연산 그래프가 실행 시점에 구성되어 모델을 유연하게 작성하고 디버깅하기 쉽다.
- **자동 미분(Autograd)**: forward 연산 기록을 바탕으로 역전파에 필요한 기울기를 자동 계산한다.

```text
입력 → forward → loss 계산 → loss.backward() → gradient 계산
     → optimizer.step() → 가중치 업데이트
```

## 3.12 PyTorch 주요 모듈

### `torch`와 텐서

`torch.Tensor`는 PyTorch의 기본 자료 구조다. `torch.tensor()`는 Python 리스트나 NumPy 배열 등의 값으로부터 텐서를 생성하는 함수다.

`torch.device`는 텐서를 계산할 장치를 표현한다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([1, 2, 3], device=device)
```

### 1. `torch.nn`: 신경망 계층과 모델 구성

- `nn.Module`: 모든 신경망 모델과 계층의 기본 클래스. 보통 `__init__()`에서 계층을 만들고 `forward()`에서 데이터 흐름을 정의한다.
- `nn.Linear`: 완전 연결 선형 계층
- `nn.Embedding`: 정수 토큰 ID를 밀집 벡터로 변환하는 계층
- `nn.LayerNorm`, `nn.RMSNorm`: Transformer 학습을 안정화하는 정규화 계층
- `nn.Dropout`: 훈련 중 활성값 일부를 무작위로 0으로 만들어 과적합을 완화하는 계층
- `nn.CrossEntropyLoss`: 다음 토큰 예측과 다중 클래스 분류에 주로 사용하는 손실 함수
- `nn.MSELoss`: 연속값 회귀에 주로 사용하는 평균 제곱 오차 손실

### 2. `torch.nn.functional`: 함수형 연산

일반적으로 별도 모듈 객체를 만들 필요가 없는 함수형 연산을 제공한다. 다만 “항상 학습 가능한 파라미터가 없는 함수만 모아둔 곳”이라기보다는, 필요한 가중치를 인자로 직접 받을 수 있는 저수준 함수형 인터페이스로 이해하는 편이 정확하다.

- 활성화: `F.relu`, `F.gelu`, `F.silu`
- 확률 변환: `F.softmax`, `F.log_softmax`
- attention: `F.scaled_dot_product_attention`

`F.scaled_dot_product_attention`은 환경에 따라 FlashAttention, 메모리 효율 attention 또는 수학 구현 중 적절한 커널을 선택할 수 있다. 항상 특정 고속 커널이 사용된다고 보장되지는 않는다.

### 3. `torch.optim`: 최적화 알고리즘

- `optim.AdamW`: Adam과 weight decay를 분리한 옵티마이저로 Transformer 학습에서 널리 사용된다.
- `optim.Adam`, `optim.SGD`: 범용 딥러닝 최적화 알고리즘
- `optim.lr_scheduler`: 학습률을 단계별로 조절하는 스케줄러. `CosineAnnealingLR`, `LinearLR` 등이 있다.

LLM에서는 초반에 학습률을 서서히 키우는 warmup과 이후 학습률을 줄이는 decay를 함께 구성하는 경우가 많다.

### 4. `torch.autograd`: 자동 미분

`requires_grad=True`인 텐서가 참여한 연산을 추적하고 `backward()` 호출 시 미분값을 계산한다. 옵티마이저는 계산된 gradient를 이용해 `nn.Parameter`를 갱신한다.

### 5. `torch.utils.data`: 데이터 파이프라인

- `Dataset`: 데이터를 불러오고 인덱싱하는 인터페이스. map-style Dataset은 일반적으로 `__getitem__()`과 `__len__()`을 구현한다.
- `IterableDataset`: 스트리밍처럼 순차적으로 샘플을 생성할 때 사용한다.
- `DataLoader`: 샘플을 섞고 미니배치로 묶으며, `num_workers`를 통해 CPU 프로세스에서 병렬로 준비할 수 있는 반복자다.

`DataLoader`가 데이터를 GPU에 직접 공급한다고 단정하기보다는 CPU에서 배치를 준비하고, 학습 루프에서 `.to(device)`를 호출해 GPU로 옮기는 구조로 이해하는 것이 정확하다.

### 6. `torch.cuda`와 `torch.amp`: 하드웨어 가속

- `torch.cuda.is_available()`: CUDA 사용 가능 여부 확인
- `torch.cuda.empty_cache()`: PyTorch 캐시 할당자에서 현재 사용하지 않는 캐시 메모리를 해제한다. 살아 있는 텐서의 메모리를 해제하거나 PyTorch가 사용할 수 있는 총 메모리를 늘려주는 기능은 아니다.
- `torch.amp.autocast`: 연산별로 FP16/BF16 등의 낮은 정밀도를 선택하는 Automatic Mixed Precision 기능
- `torch.amp.GradScaler`: FP16 학습에서 작은 gradient가 0이 되는 underflow를 완화

혼합 정밀도는 일반적으로 속도와 메모리 효율을 개선하지만, 모델·장치·옵티마이저 상태에 따라 절감 폭이 달라지므로 GPU 메모리가 항상 절반으로 줄어든다고 보장할 수 없다.

### 7. `torch.distributed`: 분산 학습

- **DDP(DistributedDataParallel)**: 각 GPU에 모델 복제본을 두고 서로 다른 미니배치를 처리한 뒤 gradient를 동기화한다. 모델 전체가 각 GPU 메모리에 들어가야 한다.
- **FSDP(FullyShardedDataParallel)**: 모델 파라미터, gradient, optimizer state를 여러 worker에 나누어 저장해 GPU별 메모리 사용량을 줄인다.

대규모 학습에서는 데이터 병렬화 외에도 tensor parallelism, pipeline parallelism, activation checkpointing 등을 함께 사용할 수 있다.

## 3.13 간단한 Decoder-only 블록 예시

다음 코드는 개념을 보여주는 최소 예시다. 실제 대규모 학습에서는 효율적인 attention 커널, KV cache, RoPE, 분산 처리와 다양한 안정화 기법이 추가된다.

```python
import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch, sequence, self.n_heads, self.head_dim
            ).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch, sequence, d_model
        )
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm + residual connection
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x
```

## 참고 자료

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch 공식 문서](https://docs.pytorch.org/docs/stable/index.html)
- [PyTorch scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch FullyShardedDataParallel](https://docs.pytorch.org/docs/stable/fsdp.html)
