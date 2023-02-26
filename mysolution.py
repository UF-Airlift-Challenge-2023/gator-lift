from airlift.solutions import Solution
from airlift.envs import ActionHelper
from airlift.envs.airlift_env import ObservationHelper as oh, ActionHelper, NOAIRPORT_ID
from airlift.envs import PlaneState
import networkx as nx
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
#import cuopt
import requests
import pandas as pd
from math import floor

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
            data["new_route"] = [x+1 for x in data["route"]]
            routes_and_types = {key:data[key] for key in ["new_route", "type"]}
            print("Route: \n", pd.DataFrame(routes_and_types))
    else:
        print("Error: ", res["error"])
    print("\n======================= End ===============================\n")
    

ip = "6801-70-171-49-181.ngrok.io"
port = "80"
url = "https://" + ip + "/cuopt/"

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

        clear_request = requests.delete(
            url + "clear_optimization_data", params=None, timeout=30
        )
        clear_request.json()
        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)
        self.new_change = True
        self.new_state = self.get_state(obs)

    def policies(self, obs, dones):
        if(self.new_change):
            self.solver_response = self.process_state(obs)
            self.new_state = self.get_state(obs)
            self.new_change = False

        # state = self.get_state(obs)
        # for type in state["route_map"]:
        #     for node in list(dict(state["route_map"][type].adj).keys()):
        #         for connection in list(dict(state["route_map"][type].adj[node]).keys()):
        #             if(state["route_map"][type].adj[node][connection]["mal"]!=0):
        #                 print(node, connection, state["route_map"][type].adj[node][connection]["mal"])
        
        
        my_action = self.process_response(obs)
        
        # Use the acion helper to generate an action

        # return None
        random_action = self._action_helper.sample_valid_actions(obs)
        # 'a_0' : {'process': 0, 'cargo_to_load': [], 'cargo_to_unload': [], 'destination': 0}
        return my_action

    def process_response(self, obs):
        actions = {}
        for a in obs:
            process = 0
            cargo_to_load = []
            cargo_to_unload = []
            destination = 0
            if(a in self.solver_response["vehicle_data"]):
                if(len(self.solver_response["vehicle_data"][a]["route"])>1):
                    if(self.solver_response["vehicle_data"][a]["route"][1]+1==obs[a]["current_airport"]):
                        print(a + " AT DESTINATION: " + str(obs[a]["current_airport"]))
                        self.solver_response["vehicle_data"][a]["type"] = self.solver_response["vehicle_data"][a]["type"][1:]
                        self.solver_response["vehicle_data"][a]["route"] = self.solver_response["vehicle_data"][a]["route"][1:]

                route_locations = [x+1 for x in self.solver_response["vehicle_data"][a]["route"]]
                type_locations = self.solver_response["vehicle_data"][a]["type"]
                task_ids = self.solver_response["vehicle_data"][a]["task_id"]
                task_types = self.solver_response["vehicle_data"][a]["task_type"]
                if(obs[a]["state"] == PlaneState.WAITING):
                    process = 1

                if((type_locations[0] == "Task" or type_locations[0] == "End") and (type_locations.count("Task")+type_locations.count("End"))==len(task_types)):
                    cargo_to_load = []
                    for cargo_id, task_type in list(zip(task_ids[0], task_types[0])):
                        if(task_type == "Pickup"):
                            cargo_to_load.append(cargo_id)
                        else:
                            cargo_to_unload.append(cargo_id)

                    cargo_finished = True
                    
                    for cargo in cargo_to_load:
                        if(cargo not in obs[a]["cargo_onboard"]):
                            process = 1
                            cargo_finished = False
                            break

                    for cargo in cargo_to_unload:
                        if(cargo in obs[a]["cargo_onboard"]):
                            process = 1
                            cargo_finished = False
                            break
                        

                    if(cargo_finished):
                        print(a + " LOADED AND UNLOADED: " + str(cargo_to_load) + " " + str(cargo_to_unload))
                        state = self.get_state(obs)
                        # print(state["active_cargo"])
                        task_ids.pop(0)
                        task_types.pop(0)

                # elif(type_locations[0] in ["Start", "w"]):
                if(len(route_locations)>1):
                    destination = route_locations[1]
                # elif(len(route_locations)==1):
                #     print(a + " AT DESTINATION: " + str(obs[a]["current_airport"]))
                #     self.solver_response["vehicle_data"][a]["type"] = self.solver_response["vehicle_data"][a]["type"][1:]
                #     self.solver_response["vehicle_data"][a]["route"] = self.solver_response["vehicle_data"][a]["route"][1:]

            actions[a] = {"process": process,
                            "cargo_to_load": cargo_to_load,
                            "cargo_to_unload": cargo_to_unload,
                            "destination": destination}
        return actions
        #     actions[a] = {"process": self._choice([0, 1]),
        #                   "cargo_to_load": self._sample_cargo(obs["cargo_at_current_airport"]),
        #                   "cargo_to_unload": self._sample_cargo(obs["cargo_onboard"]),
        #                   "destination": self._choice([NOAIRPORT_ID] + list(obs["available_routes"]))}
        # return actions

    def process_state(self, obs):
        state = self.get_state(obs)
        self.state = state
        self.multidigraph = oh.get_multidigraph(state)
        task_locations = []#[0]
        delivery_pairs = []
        demand = []#[0]
        task_ids = []
        task_time_windows = []#[[0,100000]]
        processing_time = []
        penalties = []
        for cargo in state["active_cargo"]:
            if(cargo.is_available):
                if(cargo.location!=0):
                    location = len(task_locations)
                    task_locations.append(cargo.location-1)
                    destination = len(task_locations)
                    task_locations.append(cargo.destination-1)
                    delivery_pairs.append([location,destination])
                    task_time_windows.append([cargo.earliest_pickup_time, cargo.soft_deadline])
                    task_time_windows.append([cargo.earliest_pickup_time, cargo.soft_deadline])
                    demand.append(cargo.weight)
                    demand.append(-cargo.weight)
                    task_ids.append(str(cargo.id))
                    task_ids.append(str(cargo.id))
                    processing_time.append(state["scenario_info"][0].processing_time)
                    processing_time.append(state["scenario_info"][0].processing_time)
                    penalties.append(100)
                    penalties.append(100)
        
        self.cargo_assignments = {a: None for a in self.agents}
        self.path = {a: None for a in self.agents}
        self.whole_path = {a: None for a in self.agents}
        self.plane_type_waypoints = {}

        self.fleet_data = {"capacities":[],"vehicle_locations":[],"vehicle_types": [], "vehicle_ids":[], "drop_return_trips" : []}

        for agent in state["agents"]:
            if(state["agents"][agent]["state"] in [PlaneState.READY_FOR_TAKEOFF, PlaneState.WAITING]):
                self.fleet_data["capacities"].append(state["agents"][agent]["max_weight"])
                starting_position = state["agents"][agent]["current_airport"]-1
                if(state["agents"][agent]["destination"]!=0):
                    starting_position = state["agents"][agent]["destination"]-1
                self.fleet_data["vehicle_locations"].append([starting_position,state["agents"][agent]["destination"]])
                self.fleet_data["vehicle_types"].append(state["agents"][agent]["plane_type"])
                self.fleet_data["vehicle_ids"].append(agent)
                self.fleet_data["drop_return_trips"].append(True)

        # sort dictionary based on vehicle types
        self.fleet_data["vehicle_types"], self.fleet_data["vehicle_locations"], self.fleet_data["capacities"], self.fleet_data["vehicle_ids"], self.fleet_data["drop_return_trips"] = zip(*sorted(zip(self.fleet_data["vehicle_types"], self.fleet_data["vehicle_locations"], self.fleet_data["capacities"], self.fleet_data["vehicle_ids"], self.fleet_data["drop_return_trips"])))
        self.fleet_data["capacities"] = [self.fleet_data["capacities"]]
        self.fleet_data["min_vehicles"] = 1

        self.waypoints = {}
        for plane_type in state["plane_types"]:
            self.waypoints[plane_type.id] = {"edges": [], "offsets": [], "weights": []}

        nodes = list(dict(self.multidigraph.adj).keys())
        print("  ", end =" ")
        for node in nodes:
            print("{:>3}".format(node), end =" ")
        print()

        for node in nodes:
            print("{:>3}".format(node), end =" ")
            connections = list(dict(self.multidigraph.adj[node]).keys())
            for plane_type_id in list(self.waypoints.keys()):
                self.waypoints[plane_type_id]["offsets"].append(len(self.waypoints[plane_type_id]["edges"]))
            # size of weights is length of nodes.
            weights = [0]*len(nodes)

            for connection in connections:
                conn_by_plane_type = self.multidigraph.adj[node][connection]
                for plane_type_id in list(self.waypoints.keys()):
                    if(plane_type_id in conn_by_plane_type):
                        weight = (self.multidigraph.adj[node][connection][plane_type_id]["time"]+state["scenario_info"][0].processing_time)
                        if not self.multidigraph.adj[node][connection][plane_type_id]["route_available"]:
                            weight += state["route_map"][type].adj[node][connection]["mal"]
                        weights[connection-1] = weight
                        self.waypoints[plane_type_id]["edges"].append(connection-1)
                        self.waypoints[plane_type_id]["weights"].append(weight)
            
            for weight in weights:
                # print with 5 characters
                # and right align it
                print("{:>3}".format(weight), end =" ")
            print()

        for plane_type_id in list(self.waypoints.keys()):
            self.waypoints[plane_type_id]["offsets"].append(len(self.waypoints[plane_type_id]["edges"]))

        self.waypoint_graph = {
            "waypoint_graph":self.waypoints
        }

        self.task_data = {
            "task_locations": task_locations,
            "demand": [demand],
            "task_time_windows": task_time_windows,
            "pickup_and_delivery_pairs": delivery_pairs,
            "task_ids": task_ids,
            # "service_time": processing_time,
            "penalties": penalties
        }

        self.fleet_data = self.fleet_data

        # print(self.waypoint_graph)
        # print(self.task_data)
        # print(self.fleet_data)

        matrix_response = requests.post(
            url + "set_cost_waypoint_graph", params=data_params, json=self.waypoint_graph
        )
        # print(f"\nWAYPOINT GRAPH ENDPOINT RESPONSE: {matrix_response.json()}\n")

        fleet_response = requests.post(
            url + "set_fleet_data", params=data_params, json=self.fleet_data
        )
        # print(f"FLEET ENDPOINT RESPONSE: {fleet_response.json()}\n")

        task_response = requests.post(
            url + "set_task_data", params=data_params, json=self.task_data
        )

        # print(f"TASK ENDPOINT RESPONSE: {task_response.json()}\n")


        solver_config = {"time_limit": 10, "number_of_climbers": 256, 
        "objectives": {
            "vehicle" : 0,
            "cost" : 1,
            "travel_time" : 0,
            "cumul_package_time" : 4,
            "cumul_earliest_diff" : 0,
            "variance_route_size" : 4,
            "variance_route_service_time" : 0
        },
        "solution_scope":0
        }

        solver_config_response = requests.post(
            url + "set_solver_config", params=data_params, json=solver_config
        )
        # print(f"SOLVER CONFIG ENDPOINT RESPONSE: {solver_config_response.json()}\n")

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
        
        # Try multiple times.
        if solver_response.status_code != 200:
            for _ in range(3):
                solver_response = requests.get(
                    url + "get_optimized_routes", params=solve_parameters, timeout=30
                )
                if solver_response.status_code == 200:
                    break
        try:
            response = solver_response.json()["response"]["solver_response"]
        except:
            print(solver_response)

        for vehicle in response["vehicle_data"]:
            response["vehicle_data"][vehicle]["task_id"] = [floor(x/2) for x in response["vehicle_data"][vehicle]["task_id"]]
        show_results(response)

        for vehicle in response["vehicle_data"]:
            # response["vehicle_data"][vehicle]["task_id"] = [floor(x/2) for x in response["vehicle_data"][vehicle]["task_id"]]
            # print(response["vehicle_data"][vehicle]["task_id"])
            
            # pickup_locations = []
            # dropoff_locations = []
            cargo_dict = {}
            for cargo in state["active_cargo"]:
                cargo_dict[cargo.id] = [cargo.location, cargo.destination]
            # print(cargo_dict)
            picked_up = []
            locations = []
            for task_id in response["vehicle_data"][vehicle]["task_id"]:
                if task_id not in picked_up:
                    picked_up.append(task_id)
                    locations.append(cargo_dict[task_id][0])
                else:
                    locations.append(cargo_dict[task_id][1])

            # Create array from locations with number of same number in a row.
            offsets = []
            for i in range(len(locations)):
                if i == 0:
                    offsets.append(1)
                elif locations[i] == locations[i-1]:
                    offsets[-1] += 1
                else:
                    offsets.append(1)

            grouped_tasks = []
            grouped_task_types = []
            for offset in offsets:
                grouped_tasks.append(response["vehicle_data"][vehicle]["task_id"][:offset])
                grouped_task_types.append(response["vehicle_data"][vehicle]["task_type"][:offset])
                response["vehicle_data"][vehicle]["task_id"] = response["vehicle_data"][vehicle]["task_id"][offset:]
                response["vehicle_data"][vehicle]["task_type"] = response["vehicle_data"][vehicle]["task_type"][offset:]

            response["vehicle_data"][vehicle]["task_id"] = grouped_tasks
            response["vehicle_data"][vehicle]["task_type"] = grouped_task_types
            
        # print(f"SOLVER RESPONSE: {response}\n")

        return response

