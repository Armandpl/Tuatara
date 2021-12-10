import math

import gym
import wandb

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

THROTTLE = 0.75
STEERING_GAIN = 4

conf = {
    "max_cte": 5.0,
    "log_level": 20,
    "car_config": {
        "body_style": "car01",
        "body_rgb": [49, 101, 240],  # subaru's color
        "car_name": "tuatara",
        "font_size": 32,
    },
    "cam_config": {
        "img_w": 224,
        "img_h": 224,
        "fov": 60,
    },
    "cam_resolution": (224, 224, 3)
}

track = "donkey-roboracingleague-track-v0"


def custom_reward(self, done):
    if done:
        return -1.0

    if self.cte > self.max_cte:
        return -1.0

    if self.hit != "none":
        return -2.0

    reward = (1.0 - (math.fabs(self.cte) / self.max_cte)) * self.speed

    # going fast close to the center of lane yeilds best reward
    return reward


def make_env():
    env = gym.make(track, conf=conf)
    env.set_reward_fn(custom_reward)
    env = Monitor(env)  # record stats such as returns
    return env


with wandb.init(project="racecar", config=conf, sync_tensorboard=True,
                job_type="train", entity="wandb") as run:

    env = DummyVecEnv([make_env])
    model = SAC("CnnPolicy", env, buffer_size=50000,
                verbose=2, tensorboard_log=f"runs/{run.id}")
    #
    model.load("sac_vroum")
    model.learn(total_timesteps=250000)

# Exit the scene
env.close()
