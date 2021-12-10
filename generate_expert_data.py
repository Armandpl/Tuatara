import os
import pickle

import gym
import torch
import wandb
from imitation.data import rollout
from stable_baselines3.common.vec_env import DummyVecEnv


from utils.utils import resize_to_square
from utils.xy_dataset import preprocess
from models.RoadRegression import RoadRegression

THROTTLE = 0.5
STEERING_GAIN = 6

conf = {
    "max_cte": 15.0,
    "log_level": 20,
    "car_config": {
        "body_style": "car01",
        "body_rgb": [49, 101, 240],  # subaru's color
        "car_name": "tuatara",
        "font_size": 32,
    },
}

track = "donkey-roboracingleague-track-v0"

env = gym.make(track, conf=conf)


with wandb.init(project="racecar",
                job_type="eval", entity="wandb") as run:

    artifact = run.use_artifact('wandb/racecar/model:latest', type='model')
    artifact_dir = artifact.download()

    model = RoadRegression().eval().cuda(device=0)

    model.model.load_state_dict(torch.load(
        os.path.join(artifact_dir, 'model.pth')
    ))

    class RegressionPolicy:
        def __init__(self) -> None:
            pass

        def predict(self, obs, deterministic):
            obs = resize_to_square(obs, 224)
            img = preprocess(obs)
            output = model(img).squeeze()  # .detach().cpu().numpy().flatten()
            x, _ = float(output[0]), float(output[1])
            steering = x * STEERING_GAIN  # *(y+1)/2
            return [steering, THROTTLE]

    regressionPolicy = RegressionPolicy()

    venv = DummyVecEnv([lambda: env])

    terminate = rollout.min_timesteps(n=1500)
    data = rollout.generate_trajectories(regressionPolicy, venv, terminate)

    pickle.dump(data, open("expert_data/save.pkl", "wb"))
    transitions = rollout.flatten_trajectories(data)
    print(transitions.obs.shape, transitions.acts.shape)
