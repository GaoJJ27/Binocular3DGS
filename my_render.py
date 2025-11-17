#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import json

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    json_file_path = os.path.join(model_path, "cameras.json")
    # 2. 加载 JSON 文件
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)  # json_data 是列表类型（存储多个字典）

    # 3. 提取前8个字典的 img_name（处理边界情况：若不足8个则取全部）
    img_names = []
    # 截取前8个元素（切片操作：[:8] 会自动处理“不足8个”的情况）
    for item in json_data[:8]:
        # 可选：跳过没有 img_name 字段的字典，避免报错
        if "img_name" in item:
            img_names.append(item["img_name"])
        else:
            print(f"警告：第 {len(img_names)+1} 个字典中未找到 'img_name' 字段")
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # print("[DEBUG] Rendering view:", idx)
        # print("[DEBUG] view.original_image.shape:", view.original_image.shape)
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{img_names[idx]}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{img_names[idx]}.png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            #  print("[DEBUG]scene.getTestCameras():", scene.getTestCameras())
            #  print("[DEBUG]scene.getTrainCameras():", scene.getTrainCameras())
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--n_views", default=8, type=int)
    parser.add_argument("--dataset_name", default="Blender", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)