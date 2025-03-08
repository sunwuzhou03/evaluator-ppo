import json
import custom_envs
import gymnasium as gym
from train_agent import *
from PIL import Image
import os
import shutil
from custom_envs.tvm_v3.tvm_v3 import TVM
from test_enet import Nne
from split_network import extract_and_save_evaluate_weights
def clear_directory(directory, delete_subfolders=False):
    """
    清除指定目录下的所有文件和（可选）子文件夹。

    :param directory: 要清除的目录路径
    :param delete_subfolders: 是否删除子文件夹，默认为False
    """
    # 检查目录是否存在
    if not os.path.exists(directory):
        print("指定的目录不存在")
        return

    # 遍历目录中的所有文件和文件夹
    for filename in os.listdir(directory):
        # 构造完整的文件路径
        file_path = os.path.join(directory, filename)
        
        # 检查这是一个文件还是文件夹
        if os.path.isfile(file_path) or os.path.islink(file_path):
            # 删除文件或链接
            os.unlink(file_path)
            print(f"已删除文件：{file_path}")
        elif os.path.isdir(file_path):
            if delete_subfolders:
                # 删除文件夹及其内容
                shutil.rmtree(file_path)
                print(f"已删除文件夹及其内容：{file_path}")
            else:
                print(f"子文件夹未被删除：{file_path}")

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(14, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(14, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 13), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, 13))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_action(self, x):
        action_mean = self.actor_mean(x)
        return action_mean


evaluate_network=Nne()
nne_path="custom_envs/evaluate_network_weights_1.pth"
evaluate_network.load_state_dict(torch.load(nne_path))


def obs_pipeline(env,raw_obs,encode_mode=0):

    red_states=raw_obs['red_states']
    blue_states=raw_obs['blue_states']

    monster_state_list=red_states['monster_state_list']

    tower=blue_states['tower']
    tx,ty=tower[0],tower[1]
    t_hp,t_aa=tower[2],tower[3]

    obs=[]
    if encode_mode==0:

        monster_state_list = sorted(
            monster_state_list,
            key=lambda x: (0*abs(x[0]-5)/5-abs(x[1]) / 10 + 0*abs(x[2]) / 1 + 0*abs(x[3]) / 1+0*t_hp/3+0*t_aa/3),
            reverse=True
        )[:3]
        env.update_obs(np.array(monster_state_list).flatten())
        monster_state_list=sorted(monster_state_list,key=lambda x: x[0])
        for m_state in monster_state_list:
            m_state[0]-=tx
            m_state[1]-=ty
            obs.extend(m_state)

    elif encode_mode==1:
        # monster_state_list=sorted(monster_state_list,key=lambda x: (abs(x[1])/11-abs(x[2])/2-abs(x[3])/1)/3)[:3]
        monster_state_list = sorted(
            monster_state_list,
            key=lambda x:  1*abs(x[0]-tx)/5- 1*abs(x[1])/10 + abs(x[2]) / 1 + abs(x[3]) / 1-1*t_hp/3-1*t_aa/3,#(0*abs(x[0])/10-abs(x[1]) / 10 + abs(x[2]) / 1 + abs(x[3]) / 1),
            reverse=True
        )[:3]
        env.update_obs(np.array(monster_state_list).flatten())
        monster_state_list=sorted(monster_state_list,key=lambda x: x[0])
        for m_state in monster_state_list:
            m_state[0]-=tx
            m_state[1]-=ty

            obs.extend(m_state)
            
    elif encode_mode==2:

        for m_state in monster_state_list:
            m_state[0]-=tx
            m_state[1]-=ty
        with torch.no_grad():
            monster_num=len(monster_state_list)
            monster_feature0=torch.tensor(monster_state_list,dtype=torch.float32).reshape(-1,4)
            monster_feature1=torch.tensor(monster_state_list,dtype=torch.float32).reshape(1,monster_num,4)
            other_feature=torch.tensor(tower[2:],dtype=torch.float32).reshape(1,2).expand(monster_num, 2)
            feature=torch.cat([monster_feature0,other_feature],dim=1)
            score=-evaluate_network(feature).reshape(1,-1) #(batch,shape)->(batch,1)
            top3_values, top3_indices = torch.topk(score, k=3, dim=1)
            selected_features = torch.gather(monster_feature1, dim=1, index=top3_indices.unsqueeze(-1).expand(-1, -1, 4))
            env.update_obs(selected_features.flatten())

            sort_indices = torch.argsort(selected_features[:, :, 0], dim=1)  # 对每个 batch 的第 0 个特征值排序

            # 使用 gather 对 selected_features 进行重新排序
            sorted_features = torch.gather(
                selected_features,
                dim=1,
                index=sort_indices.unsqueeze(-1).expand(-1, -1, selected_features.size(-1))
            )

            sorted_features=sorted_features.reshape(-1,3*4)
            sorted_features_list = sorted_features.flatten().tolist()

        obs.extend(sorted_features_list)
        

    obs.extend(tower[2:])  
            
    # print(obs)
    return np.array(obs)


if __name__ == "__main__":

    # TRY NOT TO MODIFY: seeding
    args = tyro.cli(Args)
    # args.capture_video=False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env =gym.make("TVM-v3", render_mode="rgb_array")
    env = gym.wrappers.ClipAction(env)
    
    agent = Agent(env).to(device)

    record=False # 是否进行对局信息记录

    monster_num_list=[6]
    train_seed_list=[42]

    model_path_list=["runs/TVM-v1__train_agent_enetplus__1__1741156594/train_agent_enetplus_final.cleanrl_model"]
    for ii,model_path in enumerate(model_path_list):
            
        # extract_and_save_evaluate_weights(model_path.replace("final", "150"))
        r_data={}
        for monster_num in monster_num_list:
            for agent_type in [1]:


                agent.load_state_dict(torch.load(model_path))
                succ_cnt=0
                max_cnt=100
                for j in range(max_cnt):
                    done=False
                    options={'monster_num':monster_num}
                    raw_obs,info=env.reset(seed=j,options=options)

                    while not done:

                        obs=obs_pipeline(env,raw_obs,agent_type)

                        # 假设 env.render() 返回一个 RGB 图像数组
                        image_array = env.render()

                        
                        # 将图像数组转换为 Pillow 图像对象
                        image = Image.fromarray(np.array(image_array))

                        # 指定文件夹路径和文件名
                        output_folder = "./rendered_images"
                        os.makedirs(output_folder, exist_ok=True)  # 如果文件夹不存在，自动创建

                        # 文件名可以按序号或时间戳命名
                        filename = os.path.join(output_folder, f"image_{len(os.listdir(output_folder))+1}.png")

                        # 保存图像
                        image.save(filename)
                        print(f"图像已保存到: {filename}")


                        obs=torch.tensor(obs,dtype=torch.float32).reshape(1,-1).to(device)
                        action=agent.get_action(obs)
                        action=list(action.detach().cpu().numpy().reshape(-1))
                        raw_obs,reward,terminated,truncated,info=env.step(action)
                        done=terminated or truncated
                        if terminated:
                            succ_cnt+=1
                        # env.render()



        
                print("monster_num",monster_num,"agent_type",agent_type,"succ_rate: ",succ_cnt/max_cnt)
                r_data[f'monster_num_{monster_num}-agent_type_{agent_type}']=  succ_cnt/max_cnt
        
        json_str = json.dumps(r_data, indent=4, ensure_ascii=False) 
        with open(f'./results/succ_rate_{ii}.json', 'w',encoding='utf-8') as f:
            f.write(json_str)