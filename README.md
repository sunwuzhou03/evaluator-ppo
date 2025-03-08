# 环境准备

```
conda create -n e-ppo python=3.9
```

```
conda activate e-ppo
```

```
pip install -r requirements.txt
```

# 文件说明
python 版本3.9.21

pip配置环境时可以更换清华源

draft.py 草稿文件

gen_render_image.py 针对TVM-v3环境生成渲染图片

plot_exp.py 使用rl_plotter绘制rl训练的曲线图

run_exp_0.py 多次实验的脚本文件

tb2csv.py 将tensorboard的曲线文件转换成csv文件，文件保存在tbcsv下

csv2logs.py 将csv文件转换成rl_plotter绘图文件，文件保存在logs下


test_agent.py 测试智能体

test_tvm.py 测试环境能否运行

test_cuda.py 测试torch是不是cuda版本的

train_agent.py 使用规则评估器训练智能体ppo代码

train_agent_enetplus.py 训练神经网络评估器的e-ppo算法

train_agent_ep.py 使用神经网络评估器训练智能体的ppo代码

videos 训练过程中的视频

runs tensorboard存储文件夹，训练曲线，查看命令 tensorboard --logdir=./runs

custom_envs 自定义环境

cleanrl、cleanrl_utils 强化学习训练框架cleanrl

rl_plotter rl绘图包





