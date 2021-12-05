import gym
import numpy as np
from pl_bolts.models.autoencoders import VAE


class VAEFeatureExtractionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        vae = VAE(input_height=224, pretrained='imagenet',
                  input_channels=3, encoder='resnet18')

        encoder = vae.encoder
        encoder.eval()

        self.encoder = encoder
        self.observation_space = gym.spaces.Box(low=np.finfo(np.float32).min,
                                                high=np.finfo(np.float32).max,
                                                shape=(1, vae.enc_out_dim),
                                                dtype=np.float32)

    def observation(self, obs):
        obs = self.encoder(obs)
        return obs
