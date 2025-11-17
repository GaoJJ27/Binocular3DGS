import os
import subprocess
import sys

def batch_train_frames():
    # 配置根路径
    input_root = "/home/Binocular3DGS/data/val"
    output_root = "/home/Binocular3DGS/output/val"

    # 确保输出根目录存在
    os.makedirs(output_root, exist_ok=True)

    # 遍历指定 seq 目录
    for seq_dir in ['012_0_seq0']:
        seq_dir_path = os.path.join(input_root, seq_dir)
        if not os.path.isdir(seq_dir_path):
            continue
        
        # 遍历当前 seq 下的 frame_xxx 文件夹（按顺序训练）
        for frame_dir in sorted(os.listdir(seq_dir_path)):
            if not frame_dir.startswith("frame_"):
                continue
            
            # 拼接输入/输出路径
            input_frame_path = os.path.join(seq_dir_path, frame_dir)
            output_frame_path = os.path.join(output_root, seq_dir, frame_dir)
            current_task = f"{seq_dir}/{frame_dir}"
            if os.path.exists(output_frame_path):
                print(f"{output_frame_path}文件已存在，跳过")
                continue
            # 构建训练命令（-u 确保 train.py 无缓冲实时输出）
            cmd = (
                f"CUDA_VISIBLE_DEVICES=3 python -u train.py "
                f"-s {input_frame_path} "
                f"-m {output_frame_path} "
                f"-r 1 "
                f"--eval"
            )

            # 父脚本核心打印（保留关键信息，简化格式）
            print("\n" + "="*80)
            print(f"📌 开始训练：{current_task}")
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
                print(f"✅ 训练成功：{current_task}")
            else:
                print(f"❌ 训练失败：{current_task}（返回码：{process.returncode}）")
            print("-"*80 + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    batch_train_frames()
    print("🎉 所有训练任务执行完毕！")
    sys.stdout.flush()