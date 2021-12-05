import gym
from pl_bolts.models.autoencoders import VAE


class FeatureExtractionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        model1 = VAE(input_height=224, pretrained='imagenet', input_channels=3, encoder='resnet18')
        encoder = model1.encoder
        encoder.eval()
        
        self.encoder = encoder

    def observation(self, obs):
        obs = self.encoder(obs)
        return obs