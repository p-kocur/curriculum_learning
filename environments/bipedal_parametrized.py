import gymnasium as gym
from gymnasium.envs.box2d.bipedal_walker import BipedalWalker

import numpy as np

SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
OBSTACLE_HEIGHT = 30 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4 
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5

class ParamBipedalWalker(BipedalWalker):
    def __init__(self, stump_height=1, stump_distance=1, **kwargs):
        super().__init__(**kwargs)
        self.hardcore = True
        self.stump_height = stump_height
        self.stump_distance = stump_distance
    
    def _generate_terrain(self, hardcore):
        GRASS, STUMP, _STATES_ = range(3)
        state = GRASS
        velocity = 0.0 
        y = TERRAIN_HEIGHT
        counter = int(TERRAIN_STARTPAD * 10)
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = [] 

        new_terrain_step = TERRAIN_STEP / 10
        new_terrain_length = TERRAIN_LENGTH * 10
        new_terrain_grass = int(self.stump_distance * 200)  # in new steps

        for i in range(new_terrain_length):
            x = i * new_terrain_step
            self.terrain_x.append(x)

            if state == STUMP and oneshot:
                if oneshot:
                    counter = int(0.5 / new_terrain_step)
                actual_stump_height = max(np.random.normal(self.stump_height, 0.05), 0.001)
                poly = [
                    (x, y),
                    (x + counter * new_terrain_step, y),
                    (x + counter * new_terrain_step, y + actual_stump_height * OBSTACLE_HEIGHT),
                    (x, y + actual_stump_height * OBSTACLE_HEIGHT),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = new_terrain_grass
                if state == GRASS and hardcore:
                    state = STUMP
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(new_terrain_length - 1):
            poly = [
                (self.terrain_x[i], self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1]),
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
            color = (76, 255 if i % 2 == 0 else 204, 76)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (102, 153, 76)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()



if __name__ == "__main__":
    env = ParamBipedalWalker(render_mode="rgb_array")
    obs, info = env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(type(reward))
        if terminated or truncated:
            obs, info = env.reset()
    env.close()