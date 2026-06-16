import math
import numpy as np
from collections import deque
from typing import List, Tuple
from scipy.spatial import cKDTree

# --- 資料結構定義 ---

class Point:
    """一個代表三維空間點的簡單類別"""
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

class WeightedPoint(Point):
    """帶有機率權重的點，用於分群和質心計算。"""
    def __init__(self, x: float, y: float, z: float, prob: float):
        super().__init__(x, y, z)
        self.prob = prob

# --- 核心演算法 ---
def create_clusters(points: List[Point], thres: float) -> List[List[Point]]:
    """
    (高效版) 使用 KD-Tree 快速地將點集進行聚類。
    """
    if not points:
        return []

    # 1. 建立 KD-Tree 以進行超快速的鄰居搜索
    coords = np.array([[p.x, p.y, p.z] for p in points])
    tree = cKDTree(coords)
    
    clusters = []
    visited_indices = np.zeros(len(points), dtype=bool)

    for i in range(len(points)):
        if visited_indices[i]:
            continue

        # 2. 找到一個新點，開始一個新集群
        cluster_indices = []
        queue = deque([i])
        visited_indices[i] = True

        while queue:
            current_index = queue.popleft()
            cluster_indices.append(current_index)
            
            # 3. 使用 tree.query_ball_point 高效找到所有鄰居
            neighbors = tree.query_ball_point(coords[current_index], r=thres)
            
            for neighbor_idx in neighbors:
                if not visited_indices[neighbor_idx]:
                    visited_indices[neighbor_idx] = True
                    queue.append(neighbor_idx)
        
        # 根據索引建立最終的集群
        clusters.append([points[j] for j in cluster_indices])
        
    return clusters