"""找到数据集第1批的真实路径（绕过编码问题）"""
import os

root = r"F:\动作数据集"
for d in os.listdir(root):
    full = os.path.join(root, d)
    if os.path.isdir(full):
        print(f"DIR: [{d}]")
        # 看里面有什么
        for sub in os.listdir(full):
            subfull = os.path.join(full, sub)
            if os.path.isdir(subfull):
                # 看再里面一层
                inner = os.listdir(subfull)[:5]
                print(f"  SUB: [{sub}] → {inner}...")
            else:
                print(f"  FILE: [{sub}]")
