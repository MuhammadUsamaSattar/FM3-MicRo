import gymnasium
import gymnasium_env


env = gymnasium.make('gymnasium_env/SingleParticleNoCargo-v0', **{'render_mode' : 'human'})
env.reset()
while 1:
    env.step([0.2,0,0,0,0,0,0,0])