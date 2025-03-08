import torch


def extract_and_save_evaluate_weights(model_path, save_path="./evaluate_network_weights.pth"):
    """
    从指定的模型状态字典中提取 evaluate_network 的权重并保存。

    :param model_path: 模型状态字典的路径
    :param save_path: 保存 evaluate_network 权重的路径
    """
    # 加载模型的状态字典
    state_dict = torch.load(model_path)

    # 提取 evaluate_network 的权重，并去掉 "critic." 前缀，替换为 "evaluate_network."
    evaluate_weights = {name.replace("critic.", "evaluate_network."): param
                        for name, param in state_dict.items()
                        if name.startswith("critic")}

    # 保存 evaluate_network 的权重
    torch.save(evaluate_weights, save_path)
    print(f"Evaluate network weights saved to {save_path}")


if __name__ == "__main__":
    model_path = "./runs\custom_envs\TVM-v1__train_agent_enet__1__1739100191/train_agent_enet_final.cleanrl_model"
    save_path = "./evaluate_network_weights.pth"
    extract_and_save_evaluate_weights(model_path, save_path)