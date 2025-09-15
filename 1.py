import numpy as np

# 计算一条回路的长度
def path_length(dist_matrix, tour):
    n = len(tour)
    return sum(dist_matrix[tour[i], tour[(i+1) % n]] for i in range(n))

# 2-opt 局部优化（直到无改进或达到迭代上限）
def two_opt(dist_matrix, tour, max_pass=100):
    n = len(tour)
    improved = True
    pass_cnt = 0
    while improved and pass_cnt < max_pass:
        improved = False
        pass_cnt += 1
        # 经典 2-opt：反转区间 [i+1, j]
        for i in range(n - 1):
            for j in range(i + 2, n if i > 0 else n - 1):
                a, b = tour[i], tour[(i + 1) % n]
                c, d = tour[j], tour[(j + 1) % n]
                before = dist_matrix[a, b] + dist_matrix[c, d]
                after  = dist_matrix[a, c] + dist_matrix[b, d]
                if after + 1e-9 < before:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
        # 若本轮没有改进则退出
    return tour

# ==== 用内存友好的启发式替代原来的 Held-Karp DP ====
def tsp_dp(dist_matrix):
    n = len(dist_matrix)
    # 最近邻构造初始解（从 0 号城市出发，保持你原有“0 为起点”的约定）
    unvisited = set(range(1, n))
    tour = [0]
    cur = 0
    while unvisited:
        # 选最近的未访问城市
        nxt = min(unvisited, key=lambda k: dist_matrix[cur, k])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    # 2-opt 优化
    tour = two_opt(dist_matrix, tour, max_pass=100)
    # 返回“回路长度”（保持你原有 min_distance 的含义为总路长）
    return path_length(dist_matrix, tour)
