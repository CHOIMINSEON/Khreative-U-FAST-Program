# 데이터 분석과 조작을 위한 라이브러리
import pandas as pd
# 데이터와 그래프 시각화를 위한 라이브러리
import seaborn as sns
from matplotlib import pyplot as plt

# CSV 불러오기
df = pd.read_csv("../Mall_Customers.csv")  # 경로 입력

# 데이터 출력
print("데이터 미리보기:")
print(df.head())
print(f"\n데이터 크기: {df.shape}")

# 그래프의 크기를 설정(가로, 세로)
fig = plt.figure(figsize=(12, 4))

# 서브플롯을 생성
# 서브플롯은 하나의 그림안에 여러 개의 작은 그래프를 배치하는 기능
# 1행 3열로 그래프를 나누고, 첫 번째 서브플롯을 ax1에 할당
ax1 = fig.add_subplot(1, 3, 1)
# 두 번째 서브플롯을 ax2에 할당
ax2 = fig.add_subplot(1, 3, 2)
# 세 번째 서브플롯을 ax3에 할당
ax3 = fig.add_subplot(1, 3, 3)

# 첫 번째 서브플롯에 연령(Age) 열의 히스토그램 시각화
sns.histplot(df["Age"], ax=ax1)
# 두 번째 서브플롯에 연간 수익(Annual Income) 열의 히스토그램 시각화
sns.histplot(df["Annual Income (k$)"], ax=ax2)
# 세 번째 서브플롯에 지출 점수(Spending Score (1-100)) 열의 히스토그램 시각화
sns.histplot(df["Spending Score (1-100)"], ax=ax3)

plt.tight_layout()
plt.show()
