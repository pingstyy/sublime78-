from stable_baselines3 import A2C, PPO
import os
import gym
env  = gym.make("LunarLander-v2")



model_dir = "models/PPO"
logdir= "logs"

if not os.pathexists(model_dir):
    os.makedirs(model_dir)

if not os.pathexists(logdir):
    os.makedirs(logdir)


model = PPO("MlpPolicy" , env, verbose=1, tensorboard_log= logdir)
# OR load model
model_path= f"{model_dir}/models.zip"
model= PPO.load_model(model_path, env= env)

TIMESTEPS =10,000
for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{model_dir}/{TIMESTEPS*i}")

model.learn(total_timesteps=10_000)

episodes = 10
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        print(reward)

# For not random but predicted action space
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.precict()   # action , states = model.precict()
        obs, reward, done, info = env.step(action)
        print(reward)







env.reset()
for step in range(200):
    env.render()
    # env.step(env.action_space.sample())
    obs, reward, done, info = env.step(env.action_space.sample())
    print(reward)
    
    
env.close()


'''
console in root
$ tensorboard --logdir=logs

'''