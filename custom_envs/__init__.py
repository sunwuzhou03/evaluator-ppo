from gymnasium.envs.registration import register

register(
     id="TVM-v3",
     entry_point="custom_envs.tvm_v3.tvm_v3:TVM",
     max_episode_steps=200,
)

register(
     id="TVM-v0",
     entry_point="custom_envs.tvm_v0.tvm_v0:TVM",
     max_episode_steps=200,
)


register(
     id="TVM-v1",
     entry_point="custom_envs.tvm_v1.tvm_v1:TVM",
     max_episode_steps=200,
)

register(
     id="TVM-v2",
     entry_point="custom_envs.tvm_v2.tvm_v2:TVM",
     max_episode_steps=200,
)

