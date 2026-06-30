# 🚀 Progressive Resolution Scheduling Strategy for Efficient 3D Gaussian Splatting

> "3DGS, 빠르고 좋은 건 알겠는데... GPU 메모리 감당 되시나요?"
> [cite_start]고품질 실시간 렌더링을 위한 3D 가우시안 스플래팅(3DGS)의 막대한 VRAM 점유율과 연산 비용을 획기적으로 줄여주는 **점진적 해상도 스케줄링(Progressive Resolution Scheduling)** 전략입니다. [cite: 12, 13, 21]

## 💡 Why Progressive Scheduling?
[cite_start]3DGS는 학습 과정에서 미분 가능한 렌더링을 위해 전체 고해상도 이미지를 메모리에 띄워둬야 하는 치명적인 단점이 있습니다. [cite: 33] 
[cite_start]사실 학습 초기 단계의 가우시안들은 씬(Scene)의 세밀한 질감보다는 전체적인 기하학적 구조 형성에 집중합니다. [cite: 20] [cite_start]즉, 처음부터 100% 원본 해상도를 투입하는 것은 데이터 중복이자 낭비라는 뜻이죠! [cite: 19, 20]

[cite_start]본 프로젝트는 다중 스케일 데이터 커리큘럼(Multi-scale Data Curriculum)을 도입하여, 제한된 하드웨어 환경에서도 대규모 공간을 복원할 수 있는 자원 효율적인 파이프라인을 제안합니다. [cite: 16, 21, 24]

## ⚙️ Core Method

[cite_start]학습 횟수에 따라 입력 이미지의 공간 해상도를 0.25배에서 원본 해상도까지 단계적으로 확장합니다. [cite: 14] [cite_start]전체 학습 횟수 $I_{max}$에 대한 해상도 배율 $R(i)$는 다음과 같이 스케줄링됩니다. [cite: 46]

[cite_start]$$R(i)=\begin{cases}0.25,&if~0\le i<T_{1}\\ 0.5,&if~T_{1}\le i<T_{2}\\ 1.0,&if~T_{2}\le i<I_{max}\end{cases}$$ [cite: 47]

* **$T_{1} = 10,000$**: 0.25x 스케일로 기하학적 골격 신속 구성 [cite: 49]
* **$T_{2} = 20,000$**: 0.5x 스케일 적용 구간 [cite: 49]
* **Total Iterations ($I_{max}$)**: 총 30,000번 반복 학습 진행 [cite: 58]

## 📊 Key Results (Blender Synthetic Dataset)
화질은 그대로 유지하면서, 자원 다이어트는 확실하게 성공했습니다! [cite: 23, 58]

* **🔥 VRAM 다이어트:** 최대 비디오 메모리 점유량 평균 **17.5% (약 825 MiB)** 절감 [cite: 69, 80]
* **✂️ 가우시안 최적화:** 평균 가우시안 수 약 **58.7%** 감소 (초기 불필요한 과잉 생성 완벽 억제) [cite: 76, 79, 80]
* **⚡ 훈련 속도 향상:** 평균 학습 시간 약 **17.0% (80초)** 단축 (기하학 구조가 복잡한 Ship 씬에서는 무려 32.7% 단축!) [cite: 71, 75, 81]
* **👀 렌더링 품질 방어:** 고정 해상도 베이스라인 대비 PSNR 손실을 단 **0.49dB** 수준으로 억제 (시각적 차이 거의 없음) [cite: 15, 61, 76]

## 🛠️ Environment
* **OS:** Ubuntu 20.04 LTS [cite: 56]
* **CPU:** Intel Xeon Silver 4214R [cite: 57]
* **GPU:** NVIDIA RTX A6000 (48GB VRAM) [cite: 57]
