import subprocess

# 定义不同的随机种子
# seeds = [42, 123, 999, 2023,3407]

seeds=[1]*5


# 批量调用命令行命令，训练使用朴素筛选器的agent
for seed in seeds:
    command = [
        "python", "train_agent.py",
        "--seed", str(seed),
    ]
    
    print(f"Running command: {' '.join(command)}")
    
    # 执行命令
    result = subprocess.run(command, capture_output=True, text=True)
    
    # 打印命令的输出
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return Code:", result.returncode)
    print("-" * 50)

# # 运行train_simple_agent.py
# command = [
#     "python", "train_simple_agent.py",
#     "--seed", str(42),
# ]

# print(f"Running command: {' '.join(command)}")

# # 执行命令
# result = subprocess.run(command, capture_output=True, text=True)

# # 打印命令的输出
# print("STDOUT:", result.stdout)
# print("STDERR:", result.stderr)
# print("Return Code:", result.returncode)
# print("-" * 50)