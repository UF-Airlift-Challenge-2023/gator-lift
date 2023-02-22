from airlift.solutions import Solution
from airlift.envs import ActionHelper
from airlift.envs.airlift_env import ObservationHelper as oh, ActionHelper, NOAIRPORT_ID
import networkx as nx
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
# import cuopt

class MySolution(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """
    def __init__(self):
        super().__init__()

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        # Currently, the evaluator will NOT pass in an observation space or action space (they will be set to None)
        super().reset(obs, observation_spaces, action_spaces, seed)
        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

    def policies(self, obs, dones):
        self.process_state(obs)
        # Use the acion helper to generate an action
        return self._action_helper.sample_valid_actions(obs)

    def process_state(self, obs):
        state = self.get_state(obs)
        task_locations = [0]
        delivery_pairs = []
        demand = [0]
        task_time_windows = [[0,100000]]
        for cargo in state["active_cargo"]:
            location = len(task_locations)
            task_locations.append(cargo.location)
            destination = len(task_locations)
            task_locations.append(cargo.destination)
            delivery_pairs.append([location,destination])
            task_time_windows.append([cargo.earliest_pickup_time, cargo.hard_deadline])
            task_time_windows.append([cargo.earliest_pickup_time, cargo.hard_deadline])
            demand.append(cargo.weight)
        self.cargo_assignments = {a: None for a in self.agents}
        self.path = {a: None for a in self.agents}
        self.whole_path = {a: None for a in self.agents}
        self.multidigraph = oh.get_multidigraph(state)
        self.waypoints = {}

        for plane_type in state["plane_types"]:
            self.waypoints[plane_type.id] = {"edges": [], "offsets": [], "weights": []}

        nodes = list(dict(self.multidigraph.adj).keys())
        for node in nodes:
            connections = list(dict(self.multidigraph.adj[node]).keys())
            for plane_type_id in list(self.waypoints.keys()):
                self.waypoints[plane_type_id]["offsets"].append(len(self.waypoints[plane_type_id]["edges"]))
            for connection in connections:
                conn_by_plane_type = self.multidigraph.adj[node][connection]
                for plane_type_id in list(self.waypoints.keys()):
                    if(plane_type_id in conn_by_plane_type):
                        if(self.multidigraph.adj[node][connection][plane_type_id]["route_available"]):
                            self.waypoints[plane_type_id]["edges"].append(connection-1)
                            self.waypoints[plane_type_id]["weights"].append(self.multidigraph.adj[node][connection][plane_type_id]["time"])
        return state

