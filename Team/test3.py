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
    
    # 결과 출력
    if path:
        print("최단 경로:", path)
        print("총 거리:", len(path) - 1)
    else:
        print("경로를 찾을 수 없습니다.")


if __name__ == "__main__":
    main()