# 데이터 분석과 조작을 위한 라이브러리
import pandas as pd
# K-Means 알고리즘 가져오기
from sklearn.cluster import KMeans
# 반복 작업 진행 상황(progress bar)을 시각적으로 보여주는 라이브러리
from tqdm import tqdm
# 수치계산과 시각화 라이브러리
import numpy as np
import matplotlib.pyplot as plt
# 인터랙티브한 그래프를 만들 수 있는 라이브러리
import plotly.graph_objects as go
import plotly


# 모델을 학습시키는 함수 정의(클러스터 수, 데이터)
def train_model(num_cluster, data):
    # KMeans 모델을 생성(클러스터 수, 초기 중심값 설정 반복 횟수, 난수 발생기의 초기값)
    model = KMeans(
        n_clusters=num_cluster,
        n_init=10,
        random_state=1234
    )
    # 모델을 데이터에 맞춰 학습
    model.fit(data)
    # 학습된 모델 반환
    return model


# 적절한 클러스터 개수를 찾는 함수
def elbow_method(data):
    # 각 클러스터 개수에 따른 inertia을 저장할 리스트를 초기화
    # inertia: 각 데이터 포인트와 해당 클러스터 중심 간 거리의 제곱합
    inertia_list = []
    
    # 클러스터 개수를 2에서 10까지 변경하며 KMeans 모델 학습
    for i in tqdm(range(2, 11)):
        # 모델 학습
        model = train_model(i, data)
        # 학습된 모델의 inertia을 리스트에 추가
        inertia_list.append(model.inertia_)
    
    # 각 클러스터 개수에 대한 inertia 리스트를 반환
    return inertia_list


# 클러스터 수에 따른 Inertia을 시각화하는 함수(클러스터 개수에 따른 inertia 값, 시각화 대상의 열 이름)
def plot_inertia(inertia_list, target_columns):
    # 클러스터 개수(2~10)와 대응하는 관성값을 선 그래프로 시각화
    plt.plot(range(2, 11), inertia_list, marker="o", label="inertia")
    # 그래프 제목 설정(열이름 '_'로 연결)
    plt.title(f"{'_'.join(target_columns)} KMeans clustering Inertia")
    # x축
    plt.xlabel("num clusters")
    # y축
    plt.ylabel("inertia")


if __name__ == "__main__":
    # CSV 불러오기
    df = pd.read_csv("../Mall_Customers.csv")
    
    # 3개 변수로 클러스터링 K값 찾기
    target_columns = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    data = df[target_columns]
    
    # inertia 리스트 계산
    print("Elbow Method로 최적 클러스터 수 찾기 (3D)...")
    inertia_list = elbow_method(data)
    
    # 시각화
    plt.figure(figsize=(8, 5))
    plot_inertia(inertia_list, target_columns)
    plt.show()
    
    # KMeans 모델 훈련 (K=6)
    print("\nK=6로 모델 훈련 중...")
    K = 6
    model = train_model(K, data)
    labels = model.labels_
    centroids = model.cluster_centers_
    
    # 데이터에서 값을 추출하여 numpy 배열로 변환
    data_ar = data.values
    
    # 3D 산점도 생성
    trace = go.Scatter3d(
        x=data_ar[:, 0],  # x축: 나이(Age) 데이터
        y=data_ar[:, 1],  # y축: 연간 수입(Annual Income) 데이터
        z=data_ar[:, 2],  # z축: 지출 점수(Spending Score) 데이터
        mode='markers',  # 데이터 포인트를 마커로 표시
        
        # 마커의 속성 설정
        marker={
            "color": labels,  # 클러스터 레이블에 따라 색상 다르게 지정
            "size": 3,  # 마커 크기 설정
            "opacity": 1  # 마커의 불투명도 설정 (1은 불투명)
        }
    )
    
    # 그래프 레이아웃 설정
    layout = go.Layout(
        title='Age-Income-Spending Clustering',  # 그래프의 제목 설정
        scene={  # 3D 그래프의 축 제목 설정
            "xaxis": {"title": "Age"},  # x축 제목: Age
            "yaxis": {"title": "Income"},  # y축 제목: Income
            "zaxis": {"title": "Spending"},  # z축 제목: Spending
        },
        width=1000,  # 그래프의 너비 설정
        height=600  # 그래프의 높이 설정
    )
    
    # Figure 객체 생성 (데이터, 그래프 레이아웃)
    fig = go.Figure(data=[trace], layout=layout)
    
    # 3D 그래프 출력
    print("\n3D 인터랙티브 그래프 생성 중...")
    fig.show()
    
    print(f"\n클러스터링 완료! (K={K})")
