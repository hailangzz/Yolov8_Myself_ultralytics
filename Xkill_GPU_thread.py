#!/usr/bin/env python3
import os
import signal
import subprocess


def get_gpu_processes():
    """获取 GPU 进程信息：PID, 显存占用, 进程名."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )

        processes = []
        for line in result.splitlines():
            pid, name, mem = [x.strip() for x in line.split(",")]
            processes.append({"pid": int(pid), "name": name, "memory": int(mem)})
        return processes

    except subprocess.CalledProcessError:
        return []


def get_resettable_gpus():
    """返回可 reset 的 GPU 列表（排除主显示 GPU）."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,display_active", "--format=csv,noheader"], encoding="utf-8"
        )

        resettable = []
        for line in result.splitlines():
            gpu_id, display_active = [x.strip() for x in line.split(",")]

            # display_active = Enabled 表示主 GPU（用于显示）
            if display_active.lower() == "disabled":
                resettable.append(gpu_id)

        return resettable

    except subprocess.CalledProcessError:
        return []


def kill_process(pid):
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"✔ Killed PID {pid}")
    except ProcessLookupError:
        print(f"⚠ PID {pid} not found")
    except PermissionError:
        print(f"⛔ No permission to kill PID {pid}")


def gpu_reset():
    """仅 reset 非显示 GPU，避免主 GPU 报错."""
    print("\n🧹 Attempting GPU reset (fragment cleanup)...")

    gpus = get_resettable_gpus()

    if not gpus:
        print("⚠ No resettable GPU found (primary GPU in use). Skipping reset.")
        return

    for gpu_id in gpus:
        try:
            subprocess.run(["nvidia-smi", "--gpu-reset", "-i", gpu_id], check=True)
            print(f"✔ GPU {gpu_id} reset successful")
        except subprocess.CalledProcessError:
            print(f"⚠ GPU {gpu_id} reset failed")


def run_sudo_command(cmd):
    """在非交互环境下执行 sudo 命令."""
    # password = os.environ.get("SUDO_PASS")
    password = "123456"
    if not password:
        raise RuntimeError("❌ SUDO_PASS environment variable not set")

    process = subprocess.run(["sudo", "-S", *cmd], input=password + "\n", text=True, capture_output=True)

    if process.returncode != 0:
        print(f"⚠ Command failed: {' '.join(cmd)}")
        print(process.stderr.strip())
    else:
        print(f"✔ Success: {' '.join(cmd)}")


def reload_nvidia_uvm():
    print("🔧 Reloading NVIDIA UVM module...")
    run_sudo_command(["rmmod", "nvidia_uvm"])
    run_sudo_command(["modprobe", "nvidia_uvm"])


import gc

import torch


def clean_gpu():
    print("🧹 Cleaning GPU memory...")

    # 删除未引用对象
    gc.collect()

    # 清空 PyTorch 缓存
    torch.cuda.empty_cache()

    # 释放 IPC 内存
    torch.cuda.ipc_collect()

    print("✅ GPU memory cleaned")


def main():
    print("🔍 Checking GPU processes...\n")

    processes = get_gpu_processes()

    if not processes:
        print("✅ No GPU processes found.")
    else:
        print("📊 GPU Processes:")
        total_mem = 0

        for p in processes:
            print(f"PID={p['pid']:>6} | {p['name']:<20} | {p['memory']} MiB")
            total_mem += p["memory"]

        print(f"\n🧠 Total GPU memory used: {total_mem} MiB\n")

        print("🔥 Killing processes...")
        for p in processes:
            kill_process(p["pid"])

    gpu_reset()

    print("\n🎉 GPU cleanup complete.")


if __name__ == "__main__":
    main()
    reload_nvidia_uvm()
    clean_gpu()
