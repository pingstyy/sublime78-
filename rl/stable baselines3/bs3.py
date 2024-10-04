#stable baselines3 uses pytorch in backend
# pip3 install gym[box2d]

# env , model= algo, agent, observation= sensor, action= step(progress in env) 
# Action Space    ==========    discrete | continuos 


from stable_baselines3 import A2C, PPO

import gym
env  = gym.make("LunarLander-v2")

print("sample action: ", env.action_space.sample())
print("observation space shape", env.observation_space.shape)
print("sample observation ", env.observation_space.sample() )

model_A2C = A2C("MlpPolicy" , env, verbose)
model_PPO = PPO("MlpPolicy" , env, verbose)
model.learn(total_timesteps=10000)

episodes = 10
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        print(reward)







env.reset()
for step in range(200):
    env.render()
    # env.step(env.action_space.sample())
    obs, reward, done, info = env.step(env.action_space.sample())
    print(reward)
    
    
env.close()