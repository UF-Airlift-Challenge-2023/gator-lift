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
from airlift.envs import PerlinMapGenerator, PlainMapGenerator

working_capacity = 2
processing_time = 10
num_airports = 12
num_cargo = 50
num_agents = 10
num_drop_off_airports = 3
num_pick_up_airports = 3
soft_deadline_multiplier = 25
hard_deadline_multiplier = 50

num_dynamic_cargo = 5
dynamic_cargo_generation_rate = 1/100
malfunction_rate = 1 / 300
min_duration = 10
max_duration = 100
route_ratio = 2

drop_off_fraction_reachable = 0.2
pick_up_fraction_reachable = 0.2

single_plane_type = [PlaneType(id=0, max_range=3, speed=0.4, max_weight=10)]

multiple_plane_types = [PlaneType(id=0, max_range=3, speed=0.2, max_weight=20),
                        PlaneType(id=1, max_range=2, speed=0.5, max_weight=3)]

max_cycles = 5000

def create_env(showroutes=False):
    return AirliftEnv(
         world_generator=AirliftWorldGenerator(
             plane_types=multiple_plane_types,
             airport_generator=RandomAirportGenerator(
                 max_airports=num_airports,
                 processing_time=processing_time,
                 working_capacity=working_capacity,
                 make_drop_off_area=True,
                 make_pick_up_area=True,
                 num_drop_off_airports=1,
                 num_pick_up_airports=1,
                 mapgen=PerlinMapGenerator()),
             route_generator=RouteByDistanceGenerator(
                 malfunction_generator=EventIntervalGenerator(
                     malfunction_rate=malfunction_rate,
                     min_duration=min_duration,
                     max_duration=max_duration),
                 route_ratio=route_ratio),
             cargo_generator=DynamicCargoGenerator(
                 cargo_creation_rate=dynamic_cargo_generation_rate,
                 max_cargo_to_create=num_dynamic_cargo,
                 num_initial_tasks=num_cargo,
                 soft_deadline_multiplier=soft_deadline_multiplier,
                 hard_deadline_multiplier=hard_deadline_multiplier),
             airplane_generator=AirplaneGenerator(num_agents),
             max_cycles=max_cycles
         ),
         renderer=FlatRenderer(show_routes=showroutes)
    )
    # return AirliftEnv(
    #     AirliftWorldGenerator(
    #         plane_types= [PlaneType(id=0, max_range=1.0, speed=0.05, max_weight=5)],
    #         airport_generator=RandomAirportGenerator(mapgen=PerlinMapGenerator(),
    #                                                  max_airports=20,
    #                                                  num_drop_off_airports=4,
    #                                                  num_pick_up_airports=4,
    #                                                  processing_time=4,
    #                                                  working_capacity=1,
    #                                                  airports_per_unit_area=2),
    #         route_generator=RouteByDistanceGenerator(malfunction_generator=EventIntervalGenerator(1 / 300, 200, 300),
    #                                                  route_ratio=2.5),
    #         cargo_generator=DynamicCargoGenerator(cargo_creation_rate=1 / 100,
    #                                               soft_deadline_multiplier=4,
    #                                               hard_deadline_multiplier=12,
    #                                               num_initial_tasks=40,
    #                                               max_cargo_to_create=10),
    #         airplane_generator=AirplaneGenerator(10),
    #     ),
    #     renderer=FlatRenderer(show_routes=showroutes)
    # )

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