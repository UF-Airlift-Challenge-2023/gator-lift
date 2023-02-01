from mysolution import MySolution
import time
from gym import logger
from airlift.envs.plane_types import PlaneType
from airlift.envs.generators.cargo_generators import DynamicCargoGenerator
from airlift.envs.airlift_env import AirliftEnv
from airlift.envs.events.event_interval_generator import EventIntervalGenerator
from airlift.envs.generators.airplane_generators import AirplaneGenerator
from airlift.envs.generators.airport_generators import RandomAirportGenerator
from airlift.envs.generators.route_generators import RouteByDistanceGenerator
from airlift.envs.generators.map_generators import PerlinMapGenerator
from airlift.envs.generators.world_generators import AirliftWorldGenerator
import click

from airlift.envs.renderer import FlatRenderer
from airlift.solutions.baselines import ShortestPath

def create_env(showroutes=False):
    return AirliftEnv(
        AirliftWorldGenerator(
            plane_types= [PlaneType(id=0, max_range=1.0, speed=0.05, max_weight=5)],
            airport_generator=RandomAirportGenerator(mapgen=PerlinMapGenerator(),
                                                     max_airports=20,
                                                     num_drop_off_airports=4,
                                                     num_pick_up_airports=4,
                                                     processing_time=4,
                                                     working_capacity=1,
                                                     airports_per_unit_area=2),
            route_generator=RouteByDistanceGenerator(malfunction_generator=EventIntervalGenerator(1 / 300, 200, 300),
                                                     route_ratio=2.5),
            cargo_generator=DynamicCargoGenerator(cargo_creation_rate=1 / 100,
                                                  soft_deadline_multiplier=4,
                                                  hard_deadline_multiplier=12,
                                                  num_initial_tasks=40,
                                                  max_cargo_to_create=10),
            airplane_generator=AirplaneGenerator(10),
        ),
        renderer=FlatRenderer(show_routes=showroutes)
    )

frame_pause_time = 0.01
solution = MySolution()
env = create_env(True)

_done = False
obs = env.reset(seed=367)  # 365  372
solution.reset(obs, seed=459)

while not _done:
    # Compute Action
    actions = solution.policies(env.observe(), env.dones)
    obs, rewards, dones, _ = env.step(actions)  # If there is no observation, just return 0
    _done = all(dones.values())

    env.render()
    if frame_pause_time > 0:
        time.sleep(frame_pause_time)