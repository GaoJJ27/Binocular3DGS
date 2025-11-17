import cv2
import torch
import json
import os
import numpy as np
import imageio
from torchvision import transforms
from tqdm import tqdm
import trimesh
import torch.nn.functional as F
from pathlib import Path
from argparse import ArgumentParser


from colmap_loader import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
from model_selection import model_type, pre_trained_model_types, select_model
from utils import DotDict, matches_from_flow, getView2World, point_world2depth, depth2point_world, get_projected_patch_color, map_points_to_image
from ssim import SSIM_v2

def triangulate_and_dense_point_cloud(frame_name, images_list, masks_dir, extri_data, intri_data, output_folder, args):
    # 修正：正确处理灰度图和彩色图
    masks_list = []
    for mask_name in tqdm(masks_dir):
        mask_img = imageio.v3.imread(mask_name)
        # 如果是彩色图，取第一个通道；如果是灰度图，直接使用
        if mask_img.ndim == 3:
            mask_img = mask_img[..., 0]  # 取第一个通道
        # 如果是灰度图 (ndim == 2)，直接使用
        masks_list.append(mask_img)

    masks = np.stack(masks_list)
    masks = torch.tensor(masks, dtype=torch.float32, device="cuda") / 255.0

    c2w_list = []
    for key, value in extri_data.items():
        extri_data[key] = np.array(value, dtype=float)
        R = extri_data[key][:3, :3]
        t = extri_data[key][:3, 3]
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        c2w_list.append(c2w)

    # 堆叠所有 c2w 矩阵（形状为 [N, 4, 4]，N 是字典中键值对的数量）
    c2w_stack = np.stack(c2w_list, axis=0)
    extrinsics_all = torch.from_numpy(c2w_stack).float().cuda()

    # extrinsics_all = torch.tensor(list(extri_data.values())).float().cuda()
    intrinsics_all = torch.tensor(list(intri_data.values())).float().cuda()

    # 转成1000x549分辨率的内参 
    intrinsics_all = intrinsics_all / 4
    intrinsics_all[:, 2, 2] = 1.0
    intrinsics_all[:, 0, 2] = 1000/2
    intrinsics_all[:, 1, 2] = 549/2

    n_images = len(images_list)
    if args.dataset_name == "LLFF":
        llffhold=8
        train_idx = [idx for idx in range(n_images) if idx % llffhold != 0]
        idx_sub = [round(i) for i in np.linspace(0, len(train_idx) - 1, args.n_views)]
    elif args.dataset_name == "DTU":
        idx_sub = args.dtu_sparse_indices[:args.n_views]
    else:
        raise NotImplementedError(args.dataset_name)
    ref_indices = idx_sub
    srcs_indices = {}
    for idx in ref_indices:
        indices = ref_indices.copy()
        indices.remove(idx)
        srcs_indices[idx] = indices

    print("reading images ...")
    images = np.stack([imageio.v3.imread(image_name)[..., :3] for image_name in tqdm(images_list)])
    images = torch.tensor(images).float().cuda()

    image_w = 1000
    image_h = 549
    image_wh = torch.tensor([image_w - 1, image_h - 1], device="cuda")

    print("start key points prediction")
    points_3D = []
    colors_all = []
    for ref_index in ref_indices:
        depth_list = []
        coord_list = []
        for src_index in srcs_indices[ref_index]:
            ref_image = images[ref_index]
            ref_image = ref_image.cuda()
            src_image = images[src_index]
            src_image = src_image.cuda()

            with torch.inference_mode():
                pred = matcher.get_matches_and_confidence(ref_image.permute(2,0,1).unsqueeze(0),
                                                        src_image.permute(2,0,1).unsqueeze(0))
            mkpts0 = pred["kp_source"]
            mkpts1 = pred["kp_target"]
            confidence = pred["confidence_value"]
            # mask = (confidence > 0.2)
            # mkpts0 = mkpts0[mask]
            # mkpts1 = mkpts1[mask]

            ref_image_name = os.path.basename(images_list[ref_index])
            ref_image_name = ref_image_name.split(".")[0]
            src_image_name = os.path.basename(images_list[src_index])
            src_image_name = src_image_name.split(".")[0]
            print(f"key points from {ref_image_name} and {src_image_name} : {len(mkpts0)}")
            if len(mkpts0) == 0:
                continue

            # Use triangulation to get 3D points
            ref_c2w = extrinsics_all[ref_index]
            src_c2w = extrinsics_all[src_index]
            intrinsic = torch.cat([intrinsics_all[0], torch.zeros((3, 1), device="cuda")], dim=1)
            ref_p = torch.matmul(intrinsic, torch.inverse(ref_c2w))
            src_p = torch.matmul(intrinsic, torch.inverse(src_c2w))
            points = cv2.triangulatePoints(ref_p.cpu().numpy(), src_p.cpu().numpy(), mkpts0.T, mkpts1.T)
            points = (points / points[3])[:3].T

            # project 3D points to image
            points = torch.tensor(points, device="cuda")
            ref_w2c = torch.inverse(ref_c2w)
            src_w2c = torch.inverse(src_c2w)
            ref_uv, ref_depth = point_world2depth(points=points.reshape(-1, 3),
                                        intrinsic_matrix=intrinsic[:3, :3],
                                        w2c=ref_w2c)
            src_uv, src_depth = point_world2depth(points=points.reshape(-1, 3),
                                        intrinsic_matrix=intrinsic[:3, :3],
                                        w2c=src_w2c)

            ### filter ####
            mkpts0_tensor = torch.tensor(mkpts0).cuda()
            mkpts1_tensor = torch.tensor(mkpts1).cuda()
            ref_norm = torch.norm(ref_uv - mkpts0_tensor, dim=-1)
            src_norm = torch.norm(src_uv - mkpts1_tensor, dim=-1)

            mask = (ref_norm < 2.0) & (src_norm < 2.0)
            points = points[mask]
            ref_uv = ref_uv[mask]
            src_uv = src_uv[mask]
            ref_depth = ref_depth[mask]
            src_depth = src_depth[mask]
            mkpts0_tensor = mkpts0_tensor[mask]
            mkpts1_tensor = mkpts1_tensor[mask]

            uv_mask = (ref_uv[:, 0] >= 0) & (ref_uv[:, 0] <= image_w-1) & (ref_uv[:, 1] >= 0) & (ref_uv[:, 1]<=image_h-1) & \
                  (src_uv[:, 0] >= 0) & (src_uv[:, 0] <= image_w-1) & (src_uv[:, 1] >= 0) & (src_uv[:, 1]<=image_h-1)
            points = points[uv_mask]
            ref_uv = ref_uv[uv_mask]
            src_uv = src_uv[uv_mask]
            ref_depth = ref_depth[uv_mask]
            src_depth = src_depth[uv_mask]
            mkpts0_tensor = mkpts0_tensor[uv_mask]
            mkpts1_tensor = mkpts1_tensor[uv_mask]

    ##############################################
            # 使用mask过滤点云
            ref_uv_int = ref_uv.long()  # 转换为整数索引
            src_uv_int = src_uv.long()

            # 从mask中提取对应位置的值
            ref_mask_values = masks[ref_index, ref_uv_int[:, 1], ref_uv_int[:, 0]]  # 注意y, x顺序
            src_mask_values = masks[src_index, src_uv_int[:, 1], src_uv_int[:, 0]]

            # 创建mask过滤条件 (mask值>0.5表示有效区域)
            mask_filter = (ref_mask_values > 0.5) & (src_mask_values > 0.5)

            # 应用mask过滤
            points = points[mask_filter]
            ref_uv = ref_uv[mask_filter]
            src_uv = src_uv[mask_filter]
            ref_depth = ref_depth[mask_filter]
            src_depth = src_depth[mask_filter]
            mkpts0_tensor = mkpts0_tensor[mask_filter]
            mkpts1_tensor = mkpts1_tensor[mask_filter]
            ##############################################

            points_3D.append(points.cpu().numpy())

            ref_uv_normal = (ref_uv / image_wh) * 2 - 1.0
            colors = F.grid_sample(ref_image.permute(2, 0, 1).unsqueeze(0), grid=ref_uv_normal.reshape(1, -1, 1, 2),
                                align_corners=False)
            colors = colors[0, :, :, 0].permute(1, 0)
            colors = colors.cpu().numpy()
            colors_all.append(colors.astype(np.uint8))

        if args.dataset_name == "DTU":
            depth_max = 10.0
            image = images[ref_index]
            intrinsic = intrinsics_all[ref_index]
            extrinsic = extrinsics_all[ref_index]
            depth = torch.ones_like(image, device="cuda")[..., 0] * depth_max

            depth_points = depth2point_world(depth, intrinsic, torch.inverse(extrinsic))
            depth_points = depth_points.cpu().numpy()
            # depth_colors = image.reshape(-1, 3).numpy().astype(np.uint8)
            depth_colors = (torch.ones_like(image, device="cpu").reshape(-1, 3).numpy() * 255.0).astype(np.uint8)

            bg_mask = (image.max(dim=-1, keepdim=True).values >= 254).reshape(-1).cpu().numpy()
            depth_points = depth_points[bg_mask]
            depth_colors = depth_colors[bg_mask]

            points_3D.append(depth_points)
            colors_all.append(depth_colors)


    points_3D = np.concatenate(points_3D, axis=0)
    colors_all = np.concatenate(colors_all, axis=0)

    mesh = trimesh.Trimesh(vertices=points_3D, vertex_colors=colors_all)
    mesh.export(os.path.join(output_folder, f"{frame_name}.ply"))
    print("save keypoints to:", (os.path.join(output_folder, f"{frame_name}.ply")))


if __name__ == "__main__":
    
    parser = ArgumentParser(description="Triangulate script parameters")
    parser.add_argument("--network_type", type=str, default="PDCNet_plus")
    parser.add_argument("--pre_trained_model", type=str, default="megadepth")
    parser.add_argument("--multi_stage_type", type=str, choices=['d', 'h', 'ms'], default="h")
    parser.add_argument("--confidence_map_R", type=float, default=1.0)
    parser.add_argument("--ransac_thresh", type=float, default=1.0)
    parser.add_argument("--mask_type", type=str, default="proba_interval_1_above_10")
    parser.add_argument("--homography_visibility_mask", action="store_true", default=True)
    parser.add_argument("--scaling_factors", type=float, nargs="+", default=[0.5, 0.6, 0.88, 1, 1.33, 1.66, 2])
    parser.add_argument("--compute_cyclic_consistency_error", action="store_true", default=True)

    # parser.add_argument("--data_path", type=str, default="data_triangulate/val/001_1_seq0")
    parser.add_argument("--data_path", type=str, default="data_triangulate/test")
    parser.add_argument("--n_views", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=1)
    # parser.add_argument("--dtu_sparse_indices", type=int, nargs="+", default=[25, 22, 28, 40, 44, 48, 0, 8, 13])
    # parser.add_argument("--dtu_sparse_indices", type=int, nargs="+", default=[1, 4, 7, 5, 0, 2, 3, 6])
    # parser.add_argument("--dtu_sparse_indices", type=int, nargs="+", default=[0, 6, 7, 2, 3, 5, 1, 4])
    parser.add_argument("--dtu_sparse_indices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--output_path", type=str, default="keypoints_to_3d/SIGA_points")
    parser.add_argument("--dataset_name", type=str, default="DTU")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    if args.network_type not in model_type:
        raise ValueError('The model that you chose is not valid: {}'.format(args.network_type))
    if args.pre_trained_model not in pre_trained_model_types:
        raise ValueError('The pre-trained model type that you chose is not valid: {}'.format(args.pre_trained_model))
    choices_for_multi_stage_types = ['d', 'h', 'ms']
    if args.multi_stage_type not in choices_for_multi_stage_types:
        raise ValueError('The inference mode that you chose is not valid: {}'.format(args.multi_stage_type))

    global_optim_iter = 3
    local_optim_iter = 7
    path_to_pre_trained_models = '/home/Binocular3DGS/submodules/dense_matcher/pre_trained_models/'
    matcher, estimate_uncertainty = select_model(args.network_type, args.pre_trained_model, args, global_optim_iter, local_optim_iter,
                                                path_to_pre_trained_models=path_to_pre_trained_models)

    print("=======================strating==================================")
    for filename in os.listdir(args.data_path):
        data_path = os.path.join(args.data_path, filename)
        print("**************************开始处理data_path:", data_path)
    # data_path = args.data_path
        images_folder = os.path.join(data_path, "images")    # image_folder = "/home/Binocular3DGS/triangulate_data/val/001_1_seq0/images"
        masks_folder = os.path.join(data_path, "masks")     # mask_folder = "/home/Binocular3DGS/triangulate_data/val/001_1_seq0/masks"
        extri_path = os.path.join(data_path, "train_extri_P4x4.json")      #extri_path = "/home/Binocular3DGS/triangulate_data/val/001_1_seq0/train_extri_P4x4.json"
        intri_path =  os.path.join(data_path, "train_intri_K3x3.json")     #intri_path = "/home/Binocular3DGS/triangulate_data/val/001_1_seq0/train_intri_K3x3.json"
        output_folder = os.path.join(args.output_path, os.path.basename(data_path))    # keypoints_to_3d/001_1_seq0"
        os.makedirs(output_folder, exist_ok=True)

        scene_name = os.path.basename(data_path)

        with open(extri_path, "r", encoding="utf-8") as f:
            extri_data = json.load(f)
            print("Loaded extrinsics for {} images.".format(len(extri_data)))     
            print("读取外参类型为：", type(extri_data))     

        with open(intri_path, "r", encoding="utf-8") as f:
            intri_data = json.load(f)
            print("Loaded intrinsics for {} images.".format(len(intri_data)))     
            print("读取内参类型为：", type(intri_data))
        
        keys = extri_data.keys()  #keys= dict_keys(['00', '01', '02', '03', '04', '05', '06', '07'])
        # 遍历每个key
        images_list = {}
        maskdir_list = {}
        
        
        for i in range(300):   #len(os.listdir(os.path.join(images_folder, '00'))) = 100
            images_list[f"frame_{i}"] = []   #frame_0
            maskdir_list[f"frame_{i}"] = []
            for key in keys:
                # 构建每个key对应的文件夹路径
                image_folder = Path(images_folder) / str(key).zfill(2)   #image_folder = /home/Binocular3DGS/triangulate_data/val/001_1_seq0/images/08
                mask_folder = Path(masks_folder) / str(key).zfill(2)     #mask_folder = /home/Binocular3DGS/triangulate_data/val/001_1_seq0/masks/08
                
                # 检查文件夹是否存在
                if not image_folder.exists() or not mask_folder.exists():
                    print(f"文件夹 {key} 不存在，跳过...")
                    continue

                frame_name = f"{i:06d}"
                # 提取第i帧图片
                image_path = os.path.join(image_folder, f"{frame_name}.jpg")      # image_path = /home/Binocular3DGS/triangulate_data/val/001_1_seq0/images/08/000000.jpg
                mask_path = os.path.join(mask_folder, f"{frame_name}.png")        # mask_path = /home/Binocular3DGS/triangulate_data/val/001_1_seq0/masks/08/000000.png
                # print("[DEBUG]image_path:", image_path)
                # print("[DEBUG]mask_path:", mask_path)
                images_list[f"frame_{i}"].append(Path(image_path))
                maskdir_list[f"frame_{i}"].append(Path(mask_path))
            # print(f"第{i}帧images_list: ", images_list['frame_{i}'])
        # print(f"第0帧images_list: ", images_list['frame_0'])
        # print(f"第0帧maskdir_list: ", maskdir_list['frame_0'])

        print("*******************开始三角化点云...")
        print(f"*******************正在处理序列{scene_name}************************")
        for i in range(74, 300):
            print(f"*******************正在处理第{i}帧************************")
            image_list = images_list[f"frame_{i}"]
            masks_dir = maskdir_list[f"frame_{i}"]
            frame_name = f"{i:06d}"
            triangulate_and_dense_point_cloud(frame_name, image_list, masks_dir, extri_data, intri_data, output_folder, args)
        print(f"*******************序列{scene_name}处理完毕！！！************************")

