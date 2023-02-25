from airlift.solutions import Solution
from airlift.envs import ActionHelper
from airlift.envs.airlift_env import ObservationHelper as oh, ActionHelper, NOAIRPORT_ID
from airlift.envs import PlaneState
import networkx as nx
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
# import cuopt
# import requests
import pandas as pd

def show_results(res):
    print("\n====================== Response ===========================\n")
    print("Solver status: ", res["status"])
    if res["status"] == 0:
        print("Cost         : ", res["solution_cost"])
        print("Vehicle count: ", res["num_vehicles"])
        for veh_id in res["vehicle_data"].keys():
            print("\nVehicle ID: ", veh_id)
            print("----------")
            print("Tasks assigned: ", res["vehicle_data"][veh_id]["task_id"])
            data = res["vehicle_data"][veh_id]
            routes_and_types = {key:data[key] for key in ["route", "type"]}
            print("Route: \n", pd.DataFrame(routes_and_types))
    else:
        print("Error: ", res["error"])
    print("\n======================= End ===============================\n")
    

ip = "127.0.0.1"
port = "5000"
url = "http://" + ip + ":" + port + "/cuopt/"

data_params = {"return_data_state": False}


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
        self.new_change = True
        self.new_state = self.get_state(obs)

    def policies(self, obs, dones):
        if(self.new_change):
            self.solver_response = self.process_state(obs)
            self.new_change = False
            
        # Use the acion helper to generate an action

        # return None
        action = self._action_helper.sample_valid_actions(obs)
        # 'a_0' : {'process': 0, 'cargo_to_load': [], 'cargo_to_unload': [], 'destination': 0}
        return action

    def process_state(self, obs):
        state = self.get_state(obs)
        self.multidigraph = oh.get_multidigraph(state)
        if (oh.get_multidigraph(self.new_state) != self.multidigraph): #if the map changes, new change is true
            self.new_change = True
        nodes = list(dict(self.multidigraph.adj).keys())
        for nodez in nodes:
            if not nodez.airport_has_capacity():
                self.new_change = True
        task_locations = []#[0]
        delivery_pairs = []
        demand = []#[0]
        task_ids = []
        task_time_windows = []#[[0,100000]]
        for cargo in state["active_cargo"]:
            if(cargo.is_available):
                if(cargo.location!=0):
                    location = len(task_locations)
                    task_locations.append(cargo.location-1)
                    destination = len(task_locations)
                    task_locations.append(cargo.destination-1)
                    delivery_pairs.append([location,destination])
                    task_time_windows.append([cargo.earliest_pickup_time, cargo.hard_deadline])
                    task_time_windows.append([cargo.earliest_pickup_time, cargo.hard_deadline])
                    demand.append(cargo.weight)
                    demand.append(-cargo.weight)
                    task_ids.append(cargo.id)
                    task_ids.append(cargo.id)
        
        self.cargo_assignments = {a: None for a in self.agents}
        self.path = {a: None for a in self.agents}
        self.whole_path = {a: None for a in self.agents}
        self.plane_type_waypoints = {}

        self.fleet_data = {"capacities":[],"vehicle_locations":[],"vehicle_types": [], "vehicle_ids":[], "drop_return_trips" : []}

        for agent in state["agents"]:
            if(state["agents"][agent]["state"] in [PlaneState.READY_FOR_TAKEOFF, PlaneState.WAITING]):
                self.fleet_data["capacities"].append(state["agents"][agent]["max_weight"])
                self.fleet_data["vehicle_locations"].append([state["agents"][agent]["current_airport"]-1,state["agents"][agent]["destination"]])
                self.fleet_data["vehicle_types"].append(state["agents"][agent]["plane_type"])
                self.fleet_data["vehicle_ids"].append(agent)
                self.fleet_data["drop_return_trips"].append(True)

        # sort dictionary based on vehicle types
        self.fleet_data["vehicle_types"], self.fleet_data["vehicle_locations"], self.fleet_data["capacities"] = zip(*sorted(zip(self.fleet_data["vehicle_types"], self.fleet_data["vehicle_locations"], self.fleet_data["capacities"])))
        self.fleet_data["capacities"] = [self.fleet_data["capacities"]]
        self.fleet_data["min_vehicles"] = 3

        self.waypoints = {}
        for plane_type in state["plane_types"]:
            self.waypoints[plane_type.id] = {"edges": [], "offsets": [], "weights": []}

        
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
                            if self.multidigraph.adj[node][connection][plane_type_id]["time"]
        for plane_type_id in list(self.waypoints.keys()):
            self.waypoints[plane_type_id]["offsets"].append(len(self.waypoints[plane_type_id]["edges"]))

        self.waypoint_graph = {
            "waypoint_graph":self.waypoints
        }

        self.task_data = {
            "task_locations": task_locations,
            "demand": [demand],
            "task_time_windows": task_time_windows,
            "delivery_pairs": delivery_pairs,
            "task_ids": task_ids,
        }

        self.fleet_data = self.fleet_data

        print(self.waypoint_graph)
        print(self.task_data)
        print(self.fleet_data)

        matrix_response = requests.post(
            url + "set_cost_waypoint_graph", params=data_params, json=self.waypoint_graph
        )
        print(f"\nWAYPOINT GRAPH ENDPOINT RESPONSE: {matrix_response.json()}\n")

        fleet_response = requests.post(
            url + "set_fleet_data", params=data_params, json=self.fleet_data
        )
        print(f"FLEET ENDPOINT RESPONSE: {fleet_response.json()}\n")

        task_response = requests.post(
            url + "set_task_data", params=data_params, json=self.task_data
        )

        print(f"TASK ENDPOINT RESPONSE: {task_response.json()}\n")


        solver_config = {"time_limit": 10, "number_of_climbers": 128}

        solver_config_response = requests.post(
            url + "set_solver_config", params=data_params, json=solver_config
        )
        print(f"SOLVER CONFIG ENDPOINT RESPONSE: {solver_config_response.json()}\n")

        solve_parameters = {
            # Uncomment to disable/ignore constraints.

            # "ignore_capacities": True,
            # "ignore_vehicle_time_windows": True,
            # "ignore_vehicle_break_time_windows": True,
            # "ignore_task_time_windows": True,
            # "ignore_pickup_and_delivery": True,
            "return_status": False,
            "return_data_state": False,
        }

        solver_response = requests.get(
            url + "get_optimized_routes", params=solve_parameters, timeout=30
        )
        print(f"SOLVER RESPONSE: {solver_response.json()}\n")

        show_results(solver_response.json()["response"]["solver_response"])




        return state

