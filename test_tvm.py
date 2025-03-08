import custom_envs
import gymnasium as gym
import time
from test_agent import *
if __name__ == "__main__":
    env=gym.make("TVM-v0")
    done=False
    
    options={'monster_num':3}
    raw_obs,info=env.reset(seed=32,options=options)

    # obs=env.reset(seed=35)

    while not done:
        action=env.action_space.sample()
        raw_obs,reward,terminated,truncated,info=env.step(action)
        
        # exit(0)
        print(raw_obs)
        
        # obs_pipeline(env,raw_obs,encode_mode=1)

        done=terminated or truncated
        image=env.render()
        
        print(np.array(image))
