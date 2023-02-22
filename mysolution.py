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
        self.plane_type_waypoints = {}
   
        self.fleet_data = {a: None for a in self.agents}
        for a in self.fleet_data:
            self.fleet_data[a] = {}
            self.fleet_data[a]["curr_cap"] = state["agents"][a]["current_weight"]
            self.fleet_data[a]["curr_airport"]=state["agents"][a]["current_airport"]
            self.fleet_data[a]["end_airport"]=state["agents"][a]["destination"]
            self.fleet_data[a]["plane_type"]=state["agents"][a]["plane_type"]
            self.fleet_data[a]["max_weight"]=state["agents"][a]["max_weight"]

        nx.draw_networkx(self.multidigraph, with_labels = True)
        plt.show()
        for plane_type in state["plane_types"]:
            self.plane_type_waypoints[plane_type] = {}
            self.plane_type_waypoints[plane_type]["edges"] = [] 
        colors = ["blue", "orange"]
        for airplane_type in range(len(state["route_map"])):
            self.multidigraph = state["route_map"][airplane_type]
            edges = np.array(list(edge[1]-1 for edge in list(self.multidigraph.edges)))
            current_node = 0

            offsets = [0]
            for idx, edge in enumerate(list(self.multidigraph.edges)):
                if edge[0]-1 != current_node:
                    current_node = edge[0]-1
                    offsets.append(idx)

            offsets.append(len(edges))
            offsets = np.array(offsets)
            
            weights = []
            for idx, node_one in enumerate(list(dict(self.multidigraph.adj).keys())):
                for node_two in list(dict(self.multidigraph.adj)[node_one].keys()):
                    print(self.multidigraph.adj[node_one][node_two])
                    weights.append(self.multidigraph.adj[node_one][node_two]["time"])
            weights = np.array(weights)

            # w_matrix = routing.WaypointMatrix(offsets, edges, weights)

            nx.draw_networkx(self.multidigraph, with_labels = True, edge_color=colors[airplane_type])
        plt.show()
        self._full_delivery_paths = {}

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

    def policies(self, obs, dones):
        # Use the acion helper to generate an action
        return self._action_helper.sample_valid_actions(obs)

