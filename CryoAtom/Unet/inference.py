import time
import os
import sys
import torch
from scipy.spatial import cKDTree
from CryoAtom.utils.mrc_tools import load_map,make_model_grid
from CryoAtom.utils.network_tools import map_segmentation,map_reconstruction
from CryoAtom.utils.save_pdb_utils import points_to_pdb
from CryoAtom.utils.torch_utlis import get_batch_slices
from CryoAtom.Unet.Unet import SimpleUnet
from CryoAtom.Unet.atom_pick_ca import (
    WeightedPoint,
    determine_target_count,
    nms_kdtree_adaptive,
)
import numpy as np
import tqdm

def get_lattice_meshgrid_np(shape_size, no_shift=False):
    linspace = [np.linspace(
        0.5 if not no_shift else 0,
        shape - (0.5 if not no_shift else 1),
        shape,
    ) for shape in shape_size]
    mesh = np.stack(
        np.meshgrid(linspace[0], linspace[1], linspace[2], indexing="ij"),
        axis=-1,
    )
    return mesh

# def grid_to_points(grid, threshold, neighbour_distance_threshold):
#     """
#     MIT License

#     Copyright (c) 2022 Kiarash Jamali

#     This function comes from: https://github.com/3dem/model-angelo/blob/main/model_angelo/c_alpha/inference.py

#     Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
#     """
#     lattice = np.flip(get_lattice_meshgrid_np(grid.shape, no_shift=False), -1)

#     output_points_before_pruning = np.copy(lattice[grid > threshold, :].reshape(-1, 3))

#     points = lattice[grid > threshold, :].reshape(-1, 3)
#     probs = grid[grid > threshold]
#     # sorted_indices = np.argsort(probs)[::-1]
#     # probs = probs[sorted_indices]
#     # points = points[sorted_indices]
#     for _ in range(3):
#         kdtree = cKDTree(np.copy(points))
#         n = 0
#         new_points = np.copy(points)
#         for p in points:
#             neighbours = kdtree.query_ball_point(p,1.1)
#             selection = list(neighbours)
#             if len(neighbours) > 1 and np.sum(probs[selection]) > 0:
#                 keep_idx = np.argmax(probs[selection])
#                 prob_sum = np.sum(probs[selection])

#                 new_points[selection[keep_idx]] = (
#                     np.sum(probs[selection][..., None] * points[selection], axis=0)
#                     / prob_sum
#                 )
#                 probs[selection] = 0
#                 probs[selection[keep_idx]] = prob_sum

#             n += 1

#         points = new_points[probs > 0].reshape(-1, 3)
#         probs = probs[probs > 0]

#     kdtree = cKDTree(np.copy(points))
#     for point_idx, point in enumerate(points):
#         d, _ = kdtree.query(point, 2)
#         if d[1] > neighbour_distance_threshold:
#             points[point_idx] = np.nan

#     points = points[~np.isnan(points).any(axis=-1)].reshape(-1, 3)

#     output_points = points
#     return output_points, output_points_before_pruning

def grid_to_points(grid, threshold, neighbour_distance_threshold):
    lattice = np.flip(get_lattice_meshgrid_np(grid.shape, no_shift=False), -1)

    # 1. 初始萃取 (與原始版完全一致)
    output_points_before_pruning = np.copy(lattice[grid > threshold, :].reshape(-1, 3))
    output_probs_before_pruning = np.copy(grid[grid > threshold])

    points = lattice[grid > threshold, :].reshape(-1, 3)
    probs = grid[grid > threshold]
    
    # --- 新增：同步側錄 true_probs ---
    true_probs = np.copy(probs) 

    # 2. 聚類過程 (Cluster points)
    for _ in range(3):
        kdtree = cKDTree(np.copy(points))
        new_points = np.copy(points)
        
        # --- 新增：同步建立 new_true_probs ---
        new_true_probs = np.copy(true_probs)
        
        for p_idx, p in enumerate(points): # 改用 enumerate 確保索引精確
            neighbours = kdtree.query_ball_point(p, 1.1)
            selection = list(neighbours)
            if len(neighbours) > 1 and np.sum(probs[selection]) > 0:
                keep_idx_in_selection = np.argmax(probs[selection])
                keep_idx = selection[keep_idx_in_selection] # 取得在原陣列的索引
                
                prob_sum = np.sum(probs[selection])
                # --- 新增：取得該群集最大機率 ---
                cluster_max_prob = np.max(true_probs[selection])

                new_points[keep_idx] = (
                    np.sum(probs[selection][..., None] * points[selection], axis=0)
                    / prob_sum
                )
                
                probs[selection] = 0
                probs[keep_idx] = prob_sum
                
                # --- 新增：同步更新 true_probs ---
                new_true_probs[selection] = 0
                new_true_probs[keep_idx] = cluster_max_prob

        # 這一行必須與原始碼完全一致，只根據 probs > 0 過濾
        valid_indices = probs > 0
        points = new_points[valid_indices].reshape(-1, 3)
        # --- 新增：同步過濾 ---
        true_probs = new_true_probs[valid_indices]
        probs = probs[valid_indices]

    # 3. 孤立點剪枝 (完全維持原始操作順序)
    kdtree = cKDTree(np.copy(points))
    # 建立一個與當前 points 等長的 mask
    final_keep_mask = np.ones(len(points), dtype=bool)
    
    for point_idx, point in enumerate(points):
        d, _ = kdtree.query(point, 2)
        if d[1] > neighbour_distance_threshold:
            # 原始碼是設為 NaN，我們這裡記在 mask 裡
            final_keep_mask[point_idx] = False

    # 最後統一過濾，確保 points 的處理與原始 logic 數學等價
    output_points = points[final_keep_mask].reshape(-1, 3)
    # --- 新增：同步過濾機率 ---
    final_true_probs = true_probs[final_keep_mask]

    return output_points, output_points_before_pruning, final_true_probs, output_probs_before_pruning

def predict_slide(grid,model,stride: int = 100, windows_size: int = 129, batch_size: int = 1,device='cpu'):
    segmentation = map_segmentation(torch.tensor(grid, dtype=torch.float), stride=stride, windows_size=windows_size)
    segmentation = torch.stack(segmentation, dim=0)
    segmentation = segmentation[:, None]
    grid_batches = get_batch_slices(segmentation.shape[0], batch_size)
    with torch.no_grad():
        segmentation = segmentation.to(device)
        out_segmentation = torch.zeros(segmentation.shape, device=device)
        for grid_batch in grid_batches:
            out_segmentation[grid_batch] = torch.sigmoid(model(segmentation[grid_batch]))
        out_segmentation = out_segmentation[:, 0]
        out_segmentation = out_segmentation.detach().cpu().numpy()
    pred = map_reconstruction(out_segmentation, grid.shape, stride=stride, windows_size=windows_size)
    return pred

def infer(args):
    device = torch.device(args.device)
    os.makedirs(args.output_path, exist_ok=True)
    model_output_dir = os.path.join(args.output_path, "see_alpha_output_ca.cif")
    module = SimpleUnet()
    module.load_state_dict(torch.load(args.log_dir,map_location=torch.device('cpu')))

    # ==================== 🛠️ 新增：計算 SimpleUnet 參數量 ====================
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # 自動換算成百萬 (M)
    m_params = total_params / 1e6
    
    print("\n" + "="*40)
    print(f"🧠 [SimpleUnet 模型參數統計]")
    print(f"  ├─ 總參數數量 (Total): {total_params:,} ({m_params:.2f} M)")
    print(f"  └─ 可訓練參數 (Trainable): {trainable_params:,}")
    print("="*40 + "\n")
    # ================================================================

    module.to(device)
    module.eval()

    start_time = time.time() # add 
    
    if args.map_path.endswith("map") or args.map_path.endswith("mrc"):
        grid_np, voxel_size, global_origin = load_map(args.map_path)
        if args.mask_path:
            assert args.mask_path.endswith("map") or args.mask_path.endswith("mrc")
            mask_np,b1,b2 = load_map(args.mask_path)
        else:
            mask_np = np.ones(grid_np.shape)
        grid_np = grid_np*mask_np
        grid_np, voxel_size, global_origin = make_model_grid(
            np.copy(grid_np), voxel_size, global_origin, target_voxel_size=1.5
        )
    else:
        raise RuntimeError(f"File {args.map_path} is not a cryo-em density map file format.")
    grid_np = (grid_np - np.mean(grid_np)) / np.std(grid_np)
    grid = (grid_np).astype(np.float32)
    batch_size = int(args.batch_size)
    windows_size = args.windows_size
    stride = args.stride
    shape = np.array(grid_np.shape[-3:])
    total_batch_num = np.prod(np.ceil(shape/stride))

    pbar = tqdm.tqdm(
        total=total_batch_num,
        file=sys.stdout,
        position=0,
        leave=True,
    )
    if np.all(shape>windows_size):
        segmentation = map_segmentation(torch.from_numpy(grid), stride=stride, windows_size=windows_size)
        segmentation = torch.stack(segmentation, dim=0)
        segmentation = segmentation[:, None]
        grid_batches = get_batch_slices(segmentation.shape[0], batch_size)
        with torch.no_grad():
            segmentation = segmentation.to(device)
            out_segmentation = torch.zeros(segmentation.shape, device=device)
            for grid_batch in grid_batches:
                out_segmentation[grid_batch] = torch.sigmoid(module(segmentation[grid_batch]))
                pbar.update(batch_size)
            out_segmentation = out_segmentation[:, 0]
            out_segmentation = out_segmentation.detach().cpu().numpy()
        pred = map_reconstruction(out_segmentation, grid.shape, stride=stride, windows_size=windows_size)
    else:
        with torch.no_grad():
            pred = torch.sigmoid(module(torch.from_numpy(grid)[None][None].to(device)))[0,0].detach().cpu().numpy()
            pbar.update(1)
    pbar.close()

    output_ca_points, output_ca_points_before_pruning, output_ca_probs, output_ca_probs_before_pruning = grid_to_points(
        pred,threshold=args.threshold,neighbour_distance_threshold=6/np.min(voxel_size)
    )

    end_time = time.time()
    print(f"[DL_Time] : {end_time - start_time:.2f} seconds")

    # ======== 新增：TXT 輸出聚類前後資料 ========
    # 這裡計算回到物理世界的 Angstroms 坐標，以對應 PDB 的真實位置
    phys_ca_before = output_ca_points_before_pruning * voxel_size[None] + global_origin[None]
    phys_ca_after = output_ca_points * voxel_size[None] + global_origin[None]

    # 輸出：聚類前 (Before Clustering)
    with open(os.path.join(args.output_path, "CA_before_clustering.txt"), "w") as f:
        for pt, prob in zip(phys_ca_before, output_ca_probs_before_pruning):
            f.write(f"[{pt[0]}, {pt[1]}, {pt[2]}], {prob}\n")

    # 輸出：聚類後 (After Clustering)
    with open(os.path.join(args.output_path, "CA_after_clustering.txt"), "w") as f:
        for pt, prob in zip(phys_ca_after, output_ca_probs):
            f.write(f"[{pt[0]}, {pt[1]}, {pt[2]}], {prob}\n")
    # --- 輸出最原始、未過濾的機率數據 ---
    prob_txt_path = os.path.join(args.output_path, "ca_probabilities_raw.txt")
    
    # 使用 NumPy 的 ndindex 直接遍歷整個 3D 矩陣
    # 這樣保證輸出的是模型預測後「最乾淨」且「完整」的樣子
    with open(prob_txt_path, "w") as f:
        f.write("x,y,z,probability\n")
        # 取得 pred 的維度 (D, H, W)
        d, h, w = pred.shape
        for i in range(d):
            for j in range(h):
                for k in range(w):
                    # 這裡不加任何 if 判斷，直接全部輸出
                    f.write(f"{i},{j},{k},{pred[i,j,k]:.6f}\n")
            
    print(f"原始原子機率文字檔（全量輸出）已儲存至: {prob_txt_path}")
    # ============================================

    # --- 步驟2：決定最終輸出座標 ---
    # 若提供 --fasta-path，以自適應閾值 + NMS 取代 kd-tree 剪枝結果；
    # 否則沿用原本的 output_ca_points（向後相容）。
    if hasattr(args, 'use_nms') and args.use_nms and args.fasta_path:
        # 2a. 估計目標原子數
        base_target     = determine_target_count(args.fasta_path)
        target_coverage = int(base_target * args.coverage_factor)
        print(f"\n[NMS] coverage_factor={args.coverage_factor:.2f} → target_atoms={target_coverage}")

        # 2b. 直接從 pred 計算自適應閾值
        #     （與 grid_to_points 相同的起點：pred 3D 網格，不依賴中間結果）
        flat_probs = np.sort(pred[pred > 0.01].flatten())[::-1]  # 取出所有有效機率並降序排列
        if len(flat_probs) >= target_coverage:
            # 取第 target_coverage 個機率值，略微降低以增加候選點
            ca_t = float(flat_probs[min(target_coverage - 1, len(flat_probs) - 1)] * 0.95)
        else:
            ca_t = 0.1  # 候選點不足時使用最低閾值
        ca_t = max(0.0, min(1.0, ca_t * args.ca_mult))
        print(f"[NMS] 自適應閾值 CA={ca_t:.4f} (×{args.ca_mult})")

        # 2c. 直接從 pred 萃取超過 ca_t 的體素，轉為物理座標 WeightedPoint
        #     （流程與 grid_to_points 相同：lattice → mask → 物理座標換算）
        lattice_nms = np.flip(get_lattice_meshgrid_np(pred.shape, no_shift=False), -1)
        nms_mask = pred > ca_t
        nms_voxel_coords = lattice_nms[nms_mask, :].reshape(-1, 3)
        nms_probs        = pred[nms_mask]
        # 體素索引 → 物理座標（Ångström）
        nms_phys_coords  = nms_voxel_coords * voxel_size[None] + global_origin[None]
        ca_pts = [
            WeightedPoint(x=float(c[0]), y=float(c[1]), z=float(c[2]), prob=float(p))
            for c, p in zip(nms_phys_coords, nms_probs)
        ]
        print(f"[NMS] 自適應閾值過濾後點數：{len(ca_pts)}")

        # 2d. NMS 自適應聚類（取代 kd-tree 剪枝）
        ca_nms, final_radius = nms_kdtree_adaptive(
            ca_pts,
            radius=args.nms_radius,
            max_points=target_coverage,
        )
        print(f"[NMS] NMS 後保留點數：{len(ca_nms)}，最終半徑：{final_radius:.3f} Å")

        # 2e. 輸出 NMS 聚類後的 TXT（格式與 CA_after_clustering.txt 一致）
        with open(os.path.join(args.output_path, "CA_after_nms.txt"), "w") as f:
            for p in ca_nms:
                f.write(f"[{p.x}, {p.y}, {p.z}], {p.prob}\n")

        # 2f. NMS 結果作為最終 .cif 輸出
        final_ca_coords = np.array([[p.x, p.y, p.z] for p in ca_nms], dtype=np.float32)
        print(f"[Stage1 輸出] ✓ 使用 NMS 聚類結果（{len(ca_nms)} 個 CA 原子）→ see_alpha_output_ca.cif")
    else:
        # 向後相容：沿用原始 kd-tree 剪枝結果（output_ca_points 為體素索引座標，需換算）
        final_ca_coords = output_ca_points * voxel_size[None] + global_origin[None]
        print(f"[Stage1 輸出] 使用原始 kd-tree 聚類結果（{len(output_ca_points)} 個 CA 原子）→ see_alpha_output_ca.cif")
        print(f"             （若要改用 NMS，請加上 --use-nms 參數）")

    # --- 步驟3：輸出 .cif 檔案 ---
    points_to_pdb(
        os.path.join(args.output_path, "output_ca_points_before_pruning.cif"),
        output_ca_points_before_pruning * voxel_size[None] + global_origin[None],
    )
    output_file_path = os.path.join(args.output_path, "see_alpha_output_ca.cif")
    points_to_pdb(output_file_path, final_ca_coords)

    return model_output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map-path", "--v", required=True, help="input cryo-em density map"
    )
    parser.add_argument(
        "--mask-path", "--m", required=True, help="input cryo-em mask map"
    )
    parser.add_argument(
        "--output-path",
        "--o",
        required=True,
        help="The C-alpha atoms ouput path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="compute device, pick one of {cpu, cuda:number}. "
             "Default set to use cpu.",
        help="The device to carry computations on",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument(
        "--stride", type=int, default=100, help="The stride for inference"
    )
    parser.add_argument("--windows-size", type=int, default=128, help="The windows for inference")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Probability threshold for inference",
    )
    # --- atom_pick_ca NMS 聚類參數（可選）---
    parser.add_argument(
        "--fasta-path",
        type=str,
        default=None,
        help="FASTA 檔案路徑，用於估計目標原子數並啟用自適應 NMS 聚類（可選）",
    )
    parser.add_argument(
        "--coverage-factor",
        type=float,
        default=1.0,
        help="目標原子數覆蓋倍率（搭配 --fasta-path 使用，預設 1.0）",
    )
    parser.add_argument(
        "--ca-mult",
        type=float,
        default=1.0,
        help="CA 閾值乘數（預設 1.0）",
    )
    parser.add_argument(
        "--nms-radius",
        type=float,
        default=1.5,
        help="NMS 初始抑制半徑（Å），預設 1.5",
    )
    parser.add_argument("log-dir",type=str,help="The model load dir")
    args = parser.parse_args()

    infer(
        args,
    )