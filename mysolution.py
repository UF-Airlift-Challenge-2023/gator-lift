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
        state = self.get_state(obs)

        self.cargo_assignments = {a: None for a in self.agents}
        self.path = {a: None for a in self.agents}
        self.whole_path = {a: None for a in self.agents}
        self.multidigraph = oh.get_multidigraph(state)
        self.waypoints = {}

        for plane_type in state["plane_types"]:
            self.waypoints[plane_type.id] = {"edges": [], "offsets": [], "weights": []}

        for airplane_type in range(len(state["route_map"])):
            self.multidigraph = state["route_map"][airplane_type]
            self.waypoints[airplane_type]["edges"] = list(edge[1]-1 for edge in list(self.multidigraph.edges))
            
            current_node = 0
            offsets = [0]
            for idx, edge in enumerate(list(self.multidigraph.edges)):
                if edge[0]-1 != current_node:
                    current_node = edge[0]-1
                    offsets.append(idx)

            offsets.append(len(self.waypoints[airplane_type]["edges"]))
            self.waypoints[airplane_type]["offsets"] = offsets
            
            weights = []
            for idx, node_one in enumerate(list(dict(self.multidigraph.adj).keys())):
                for node_two in list(dict(self.multidigraph.adj)[node_one].keys()):
                    print(self.multidigraph.adj[node_one][node_two])
                    weights.append(self.multidigraph.adj[node_one][node_two]["time"])
            self.waypoints[airplane_type]["weights"] = weights

            # w_matrix = routing.WaypointMatrix(offsets, edges, weights)

        self.waypoint_graph = {"waypoint_graph":{}}
        self.waypoint_graph["waypoint_graph"] = self.waypoints
        print(self.waypoint_graph)
        
        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

    def policies(self, obs, dones):
        # Use the acion helper to generate an action
        return self._action_helper.sample_valid_actions(obs)

