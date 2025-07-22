import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



def map_data(data_csv):
    dt = pd.read_csv(data_csv)
    return dt

# BFS 함수
def bfs(start, targets, grid, x_max, y_max):
    visited = [[False]*(x_max+1) for _ in range(y_max+1)]
    prev = [[None]*(x_max+1) for _ in range(y_max+1)]

    queue = deque([start])
    visited[start[0]][start[1]] = True

    while queue:
        y, x = queue.popleft()

        if (y, x) in targets:
            # 목적지 도달
            path = []
            while (y, x) != start:
                path.append((x, y))  # (x, y)로 저장
                y, x = prev[y][x]
            path.append((start[1], start[0]))
            return list(reversed(path))

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 상하좌우
            ny, nx = y + dy, x + dx
            if 1 <= ny <= y_max and 1 <= nx <= x_max:
                if not visited[ny][nx] and grid[ny][nx] == 0:
                    visited[ny][nx] = True
                    prev[ny][nx] = (y, x)
                    queue.append((ny, nx))
    
    return None  # 경로 없음

def main():
    df = map_data('../dataFile/map_data.csv')
    
    # 격자 크기 정의
    x_max = df['x'].max()
    y_max = df['y'].max()
    
    # 2차원 맵 배열 생성: 기본은 0 (이동 가능), 공사현장은 1 (이동 불가)
    grid = [[0 for _ in range(x_max + 1)] for _ in range(y_max + 1)]

    # 공사현장 마킹
    for _, row in df.iterrows():
        if row.get('ConstructionSite', 0) == 1:
            grid[int(row['y'])][int(row['x'])] = 1  # y, x 순서에 주의!

    # MyHome 위치 찾기
    myhome = df[df['category'] == 3][['x', 'y']].iloc[0]
    start = (int(myhome['y']), int(myhome['x']))

    # Bandalgom Coffee 위치들 찾기 (여러 개일 수 있음)
    coffee_shops = df[df['category'] == 4][['x', 'y']]
    targets = [(int(row['y']), int(row['x'])) for _, row in coffee_shops.iterrows()]

    # 최단 경로 계산
    path = bfs(start, targets, grid, x_max, y_max)
    
    #지도 시각화 
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    #격자 그리기
    for x in range(1, x_max + 1):
        ax.axvline(x - 0.5
                   , color='lightgrey'
                   , linestyle='--'
                   , linewidth=0.5
                   )
    for y in range(1, y_max + 1):
        ax.axhline(y - 0.5
                   , color='lightgrey'
                   , linestyle='--'
                   , linewidth=0.5
                   )
        
    # 반복문을 사용하여 각 좌표에 대해 점을 그리기
    for index, row in df.iterrows():
        # 점 크기 계산 (격자 크기 비례)
        # 점 크기를 격자의 크기 비율에 맞게 설정
        size = (x_max * 10) # x값을 격자 크기에 비례하여 크기 설정

        if row['category'] == 1 or row['category'] == 2:
            plt.scatter(row['x'], row['y']
                        , color='brown'
                        , marker='o'
                        , s=size
                        )
        elif row['category'] == 3:
            plt.scatter(row['x'], row['y'], color='green'
                        , marker='^'
                        , s=size
                        )
        elif row['category'] == 4:
            plt.scatter(row['x'], row['y'], color='green'
                        , marker='s'
                        , s=size
                        )
        else:
            if row['ConstructionSite'] == 1:
                plt.scatter(row['x'], row['y'], color='gray'
                            , marker='s'
                            , s=size
                        )

    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, color='red', linewidth=2, label='최단거리')
        

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
            , markersize=10),
        Line2D([0], [0]
               , color='red'
               , lw=2
               , label='최단 경로')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.gca().invert_yaxis()

    plt.show()

    # 결과 출력
    #if path:
    #    print("최단 경로:", path)
    #    print("총 거리:", len(path) - 1)
    #else:
    #    print("경로를 찾을 수 없습니다.")


if __name__ == "__main__":
    main()