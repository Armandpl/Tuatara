import os
import statistics

import gym
import torch
import wandb

from utils.xy_dataset import preprocess
from models.RoadRegression import RoadRegression

THROTTLE = 0.75
STEERING_GAIN = 4

conf = {
    "max_cte": 15.0,
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
    }
}

track = "donkey-roboracingleague-track-v0"

env = gym.make(track, conf=conf)


def control_policy(x, y):
    steering = x * STEERING_GAIN  # *(y+1)/2
    return [steering, THROTTLE]


lap_time = float("inf")
max_cte = 0

all_lap_times = []
all_cte = []

nb_crashes = 0

with wandb.init(project="racecar",
                job_type="eval", entity="wandb") as run:

    artifact = run.use_artifact('wandb/racecar/model:latest', type='model')
    artifact_dir = artifact.download()

    model = RoadRegression().eval().cuda(device=0)

    model.model.load_state_dict(torch.load(
        os.path.join(artifact_dir, 'model.pth')
    ))

    # PLAY
    obs = env.reset()
    for i in range(1500):
        # obs = resize_to_square(obs, 224)
        img = preprocess(obs)
        output = model(img).squeeze()  # .detach().cpu().numpy().flatten()
        x, y = float(output[0]), float(output[1])
        action = control_policy(x, y)
        obs, reward, done, info = env.step(action)

        if (info["last_lap_time"]
                and info["last_lap_time"] < lap_time):  # store best lap time
            lap_time = info["last_lap_time"]

        if (info["last_lap_time"] and info["last_lap_time"]
                not in all_lap_times):  # store all lap times
            new_lap_time = info["last_lap_time"]
            all_lap_times.append(new_lap_time)

        if abs(info["cte"]) > max_cte:  # store best lap time
            max_cte = abs(info["cte"])

        all_cte.append(abs(info["cte"]))

        env.render()

        if done:
            obs = env.reset()
            nb_crashes += 1

    lap_time = round(lap_time, 2)
    if all_lap_times:
        average_lap_time = round(statistics.mean(all_lap_times), 2)
    else:
        average_lap_time = float("inf")

    max_cte = round(max_cte, 2)
    average_cte = round(statistics.mean(all_cte), 2)

    print("Best lap time:", lap_time, "seconds")
    print("Average lap time:", average_lap_time, "seconds")

    print("Max cte:", max_cte)
    print("Average cte:", average_cte)

    print("Number of crashes:", nb_crashes)

    wandb.run.summary["best_lap_time"] = lap_time
    wandb.run.summary["average_lap_time"] = average_lap_time
    wandb.run.summary["max_cte"] = max_cte
    wandb.run.summary["average_cte"] = average_cte
    wandb.run.summary["nb_crashes"] = nb_crashes


# Exit the scene
env.close()
