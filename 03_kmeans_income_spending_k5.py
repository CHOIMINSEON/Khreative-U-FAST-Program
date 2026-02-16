# 데이터 분석과 조작을 위한 라이브러리
import pandas as pd
# K-Means 알고리즘 가져오기
from sklearn.cluster import KMeans
# 반복 작업 진행 상황(progress bar)을 시각적으로 보여주는 라이브러리
from tqdm import tqdm
# 수치계산과 시각화 라이브러리
import numpy as np
import matplotlib.pyplot as plt


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


# 새로운 데이터에 대한 예측을 수행하는 함수
def inference(model, data):
    # 그리드 해상도(간격)
    h = 0.02
    # 데이터의 최소, 최대값을 구하여, 그리드 범위 설정
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    # 그리드 좌표 생성 (X, Y 값)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 그리드의 각 점에 대해 클러스터 예측
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # 예측 결과를 2D 형태로 변환
    return xx, yy, Z


# k-means 결과 시각화 함수
def plot_kmeans_cluster(xx, yy, Z, model, data, columns):
    # KMeans 모델에서 군집 레이블을 가져오기
    labels = model.labels_
    # KMeans 모델에서 각 클러스터의 중심 좌표를 가져오기
    centroids = model.cluster_centers_
    
    # 그래프 크기 설정
    plt.figure(1, figsize=(10, 5))
    # 기존 플롯 지우기
    plt.clf()
    
    # 예측된 결과 Z를 xx.shape에 맞게 reshape
    Z = Z.reshape(xx.shape)
    
    # 군집 결과를 그리드에 시각화
    plt.imshow(
        Z,
        interpolation='nearest',
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Pastel2,
        aspect='auto',
        origin='lower'
    )
    
    # 원본 데이터 포인트들을 군집에 맞는 색으로 표시
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=10)
    
    # 클러스터 중심을 강조(X좌표, Y좌표, 점크기, 색, 투명도)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', alpha=0.5)
    
    # x축과 y축 라벨을 설정
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])


if __name__ == "__main__":
    # CSV 불러오기
    df = pd.read_csv("../Mall_Customers.csv")
    
    # 연간 소득, 소비지수 클러스터링 K값 찾기
    target_columns = ["Annual Income (k$)", "Spending Score (1-100)"]
    data = df[target_columns]
    
    # inertia 리스트 계산
    print("Elbow Method로 최적 클러스터 수 찾기...")
    inertia_list = elbow_method(data)
    
    # 시각화
    plt.figure(figsize=(8, 5))
    plot_inertia(inertia_list, target_columns)
    plt.show()
    
    # KMeans 모델 훈련 (K=5)
    print("\nK=5로 모델 훈련 중...")
    K = 5
    model = train_model(num_cluster=K, data=data)
    
    # k-means 예측
    xx, yy, Z = inference(model, data.values)
    
    # 결과 시각화
    plot_kmeans_cluster(xx, yy, Z, model, data.values, target_columns)
    plt.show()
    
    print(f"\n클러스터링 완료! (K={K})")
