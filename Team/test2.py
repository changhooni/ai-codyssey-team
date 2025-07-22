import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def map_data(data_csv):
    dt = pd.read_csv(data_csv)
    return dt


df = map_data('../dataFile/map_data.csv')
#print(df.fillna(''))

# 격자 크기 정의
#x_max = 15
#y_max = 15
x_max = df['x'].max()
y_max = df['y'].max()

# 격자 좌표 생성 (x는 1부터 x_max까지, y는 1부터 y_max까지)
x = np.arange(1, x_max + 1)
#y = np.arange(y_max + 1, 1)
y = np.arange(y_max, 0, -1)

#n = len(df)
#i = 1

#while i < n:
#    x = df[i]
#    print(x)
#    i = i+1
plt.figure(figsize=(10, 10))
ax = plt.gca()

# 시각화 처리
legend_elements = [
    Line2D([0], [0]
           , marker='o'
           , color='w'
           , label='Apartment/Building'
           , markerfacecolor='brown'
           , markersize=10),
    Line2D([0], [0]
           , marker='s'
           , color='w'
           , label='Bandalgom Coffee'
           , markerfacecolor='green'
           , markersize=10),
    Line2D([0], [0]
           , marker='^'
           , color='w'
           , label='MyHome'
           , markerfacecolor='green'
           , markersize=10),
    Line2D([0], [0]
           , marker='s'
           , color='grey'
           , label='Construction Site'
           , markerfacecolor='grey'
           , markersize=10)
]
ax.legend(handles=legend_elements, loc='upper right')

#plt.figure(figsize=(8, 6))  # 그림 크기 설정
for x in range(1, x_max + 1):
    ax.axvline(x - 0.5, color='lightgrey', linestyle='--', linewidth=0.5)

for y in range(1, y_max + 1):
    ax.axhline(y - 0.5, color='lightgrey', linestyle='--', linewidth=0.5)
    


# 반복문을 사용하여 각 좌표에 대해 점을 그리기
for index, row in df.iterrows():
    # 점 크기 계산 (격자 크기 비례)
    # 점 크기를 격자의 크기 비율에 맞게 설정
    size = (x_max * 10) # x값을 격자 크기에 비례하여 크기 설정

    if row['category'] in [1, 2] and row['ConstructionSite'] == 1:
        plt.scatter(row['x'], row['y']
                    , color='gray'
                    , marker='s'
                    , s=size
                    )
    elif row['category'] in [1, 2]:
        plt.scatter(row['x'], row['y']
                    , color='brown'
                    , marker='o'
                    , s=size
                    )
    elif row['category'] == 3:
        plt.scatter(row['x'], row['y']
                    , color='green'
                    , marker='^'
                    , s=size
                    )
    elif row['category'] == 4:
        plt.scatter(row['x'], row['y']
                    , color='green'
                    , marker='s'
                    , s=size
                    )
    else:
        if row['ConstructionSite'] == 1:
            plt.scatter(row['x'], row['y']
                        , color='gray'
                        , marker='s'
                        , s=size
                       )


# 제목과 레이블 설정
#plt.title("map", fontsize=14)
#plt.xlabel("X", fontsize=12)
#plt.ylabel("Y", fontsize=12)

# 그리드 추가
#plt.grid(True)

plt.gca().invert_yaxis()

# 출력
#plt.show()

# 이미지 저장
#plt.savefig("../dataFile/map.png", dpi=300, bbox_inches='tight')


# 예시 데이터 생성
#x = np.linspace(0, 10, 100)
#y = np.sin(x)

# 플로팅
#plt.plot(x, y)

# 축 설정: y축을 반전시켜서 좌측 상단이 (1, 1)이 되도록
#plt.gca().invert_yaxis()

# x축은 기본 상태로 두기
#plt.xlim([0, 10])

# 타이틀과 레이블 추가
#plt.title("좌측 상단이 (1, 1)인 지도 시각화")
#plt.xlabel("X축")
#plt.ylabel("Y축")

# 출력
plt.show()
