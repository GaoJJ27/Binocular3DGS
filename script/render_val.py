import os
import subprocess
import sys

""" CUDA_VISIBLE_DEVICES=2 python render.py 
--model_path /home/Binocular3DGS/output/val/001_1_seq0/frame_000000 
--n_views 8 
--skip_train  
--resolution 1 
--eval 
--dataset_name Blender"""

def batch_train_frames():
    # 配置根路径
    root_dir = "/home/Binocular3DGS/output/val/012_0_seq0"
    

    # # 遍历指定 seq 目录
    # for seq_dir in ['001_1_seq0']:
    #     seq_dir_path = os.path.join(input_root, seq_dir)
    #     if not os.path.isdir(seq_dir_path):
    #         continue
        
    # 遍历当前 seq 下的 frame_xxx 文件夹（按顺序渲染）
    for frame_dir in sorted(os.listdir(root_dir)):
        if not frame_dir.startswith("frame_"):
            continue
        
        # 拼接输入/输出路径
        model_path = os.path.join(root_dir, frame_dir)
        current_task = model_path

        if os.path.exists(os.path.join(model_path, "test")):
            print(f"{frame_dir}文件已存在，跳过")
            continue

        # 构建渲染命令（-u 确保 train.py 无缓冲实时输出）
        cmd = (
            f"CUDA_VISIBLE_DEVICES=2 python -u render.py "
            f"--model_path {model_path} "
            "--skip_train "
            f"-r 1 "
            f"--eval"
        )

        # 父脚本核心打印（保留关键信息，简化格式）
        print("\n" + "="*80)
        print(f"📌 开始渲染：{current_task}")
        print(f"💻 执行命令：{cmd}")
        print("="*80)
        sys.stdout.flush()  # 确保实时显示

        # 执行命令，实时捕获并输出 train.py 的日志
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # 行缓冲，逐行实时输出
        )

        # 实时打印 train.py 的所有输出（核心聚焦）
        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                print(line, end='')  # 保持 train.py 原格式
                sys.stdout.flush()

        # 读取剩余输出（避免遗漏最后几行）
        remaining_output = process.stdout.read()
        if remaining_output:
            print(remaining_output, end='')
            sys.stdout.flush()

        # 父脚本结果打印（保留成功/失败提示）
        print("\n" + "-"*80)
        if process.returncode == 0:
            print(f"✅ 渲染成功：{current_task}")
        else:
            print(f"❌ 渲染失败：{current_task}（返回码：{process.returncode}）")
        print("-"*80 + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    batch_train_frames()
    print("🎉 所有渲染任务执行完毕！")
    sys.stdout.flush()