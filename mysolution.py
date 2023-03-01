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

        clear_request = requests.delete(
            url + "clear_optimization_data", params=None, timeout=30
        )

        self._action_helper = ActionHelper(self._np_random)

        self.current_state = {}
        self.solution = None
        self.cost_matrix = {}
        self.fleet_data = {}
        self.task_data = {}
        self.solver_config = {}
        self.solve_parameters = {}

    def policies(self, obs, dones):

        self.collect_current_status(obs)
        self.get_solution()

        random_action = self._action_helper.sample_valid_actions(obs)

        return random_action

    # def process_response(self, obs):
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

    def get_cuopt_cost_matrix(self):
        self.cost_matrix["cost_matrix"] = {}
        for plane_type_id in self.current_state["cost_graphs"]:
            self.cost_matrix["cost_matrix"][int(plane_type_id)] = self.current_state["cost_graphs"][plane_type_id]

    def get_cuopt_fleet_data(self):
        self.fleet_data["vehicle_locations"] = []
        self.fleet_data["vehicle_ids"] = []
        self.fleet_data["capacities"] = [[]]
        self.fleet_data["vehicle_types"] = []

        for agent in self.current_state["agent_info"]:
            current_agent = self.current_state["agent_info"][agent]
            self.fleet_data["vehicle_locations"].append([current_agent["start"]-1, current_agent["end"]])
            self.fleet_data["vehicle_ids"].append(agent)
            self.fleet_data["capacities"][0].append(current_agent["capacity"])
            self.fleet_data["vehicle_types"].append(current_agent["plane_type_id"])

        # Sort by vehicle_types
        self.fleet_data["vehicle_locations"] = [x for _,x in sorted(zip(self.fleet_data["vehicle_types"],self.fleet_data["vehicle_locations"]))]
        self.fleet_data["vehicle_ids"] = [x for _,x in sorted(zip(self.fleet_data["vehicle_types"],self.fleet_data["vehicle_ids"]))]
        self.fleet_data["capacities"] = [x for _,x in sorted(zip(self.fleet_data["vehicle_types"],self.fleet_data["capacities"]))]
        self.fleet_data["vehicle_types"] = sorted(self.fleet_data["vehicle_types"])



    
    def get_cuopt_task_data(self):
        self.task_data["task_locations"] = []
        self.task_data["task_ids"] = []
        self.task_data["demand"] = [[]]
        self.task_data["pickup_and_delivery_pairs"] = []
        self.task_data["task_time_windows"] = []
        self.task_data["order_vehicle_match"] = []

        for cargo in self.current_state["cargo_info"]:
            current_cargo = self.current_state["cargo_info"][cargo]

            if(current_cargo["status"] in "completed"):
                continue

            self.task_data["task_locations"].append(current_cargo["location"]-1)
            self.task_data["task_ids"].append(str(cargo)+"_pickup")
            self.task_data["demand"][0].append(current_cargo["weight"])
            if(current_cargo["status"] in "transit"):
                self.task_data["task_time_windows"].append([0, 0])
            else:
                self.task_data["task_time_windows"].append([current_cargo["pickup_time"], current_cargo["soft_deadline"]])
            if(current_cargo["carrier"]!=""):
                self.task_data["order_vehicle_match"].append({"order_id" : len(self.task_data["task_ids"])-1, "vehicle_ids":[self.fleet_data["vehicle_ids"].index([current_cargo["carrier"]])]})

            self.task_data["task_locations"].append(current_cargo["destination"]-1)
            self.task_data["task_ids"].append(str(cargo)+"_delivery")
            self.task_data["demand"][0].append(-current_cargo["weight"])
            self.task_data["task_time_windows"].append([current_cargo["pickup_time"], current_cargo["soft_deadline"]])
            if(current_cargo["carrier"]!=""):
                self.task_data["order_vehicle_match"].append({"order_id" : len(self.task_data["task_ids"])-1, "vehicle_ids":[self.fleet_data["vehicle_ids"].index([current_cargo["carrier"]])]})

            self.task_data["pickup_and_delivery_pairs"].append([len(self.task_data["task_ids"])-2, len(self.task_data["task_ids"])-1])

    def get_cuopt_solver_config(self):

        self.solver_config = {"time_limit": 10, "number_of_climbers": 256, 
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

        self.solve_parameters = {
            "return_status": False,
            "return_data_state": False,
        }
    def get_solution(self):

        # SET Cost Matrix (AKA Time Matrix)
        self.get_cuopt_cost_matrix()
        matrix_response = requests.post(
            url + "set_cost_matrix", params=data_params, json=self.cost_matrix
        )

        # SET Fleet Data
        self.get_cuopt_fleet_data()
        fleet_response = requests.post(
            url + "set_fleet_data", params=data_params, json=self.fleet_data
        )

        # SET Task Data
        self.get_cuopt_task_data()
        task_response = requests.post(
            url + "set_task_data", params=data_params, json=self.task_data
        )

        # SET Solver Config
        self.get_cuopt_solver_config()
        solver_config_response = requests.post(
            url + "set_solver_config", params=data_params, json=self.solver_config
        )

        # SOLVE
        solve_response = requests.get(
            url + "get_optimized_routes", params=data_params, json=self.solve_parameters
        )

        self.solution = solve_response.json()

    
    # def process_state(self, obs):
        # state = self.get_state(obs)
        # self.state = state
        # self.multidigraph = oh.get_multidigraph(state)

        # self.task_data = set_task_data(state)
        
        # self.cargo_assignments = {a: None for a in self.agents}
        # self.path = {a: None for a in self.agents}
        # self.whole_path = {a: None for a in self.agents}
        # self.plane_type_waypoints = {}

        # self.fleet_data = set_fleet_data(state)

        # self.waypoints = set_waypoint_data(state)
        
        # self.waypoint_graph = {
        #     "waypoint_graph":self.waypoints
        # }

        # self.fleet_data = self.fleet_data

        # matrix_response = requests.post(
        #     url + "set_cost_waypoint_graph", params=data_params, json=self.waypoint_graph
        # )
        # # print(f"\nWAYPOINT GRAPH ENDPOINT RESPONSE: {matrix_response.json()}\n")

        # fleet_response = requests.post(
        #     url + "set_fleet_data", params=data_params, json=self.fleet_data
        # )
        # # print(f"FLEET ENDPOINT RESPONSE: {fleet_response.json()}\n")

        # task_response = requests.post(
        #     url + "set_task_data", params=data_params, json=self.task_data
        # )

        # # print(f"TASK ENDPOINT RESPONSE: {task_response.json()}\n")


        # solver_config = {"time_limit": 10, "number_of_climbers": 256, 
        # "objectives": {
        #     "vehicle" : 0,
        #     "cost" : 1,
        #     "travel_time" : 0,
        #     "cumul_package_time" : 4,
        #     "cumul_earliest_diff" : 0,
        #     "variance_route_size" : 4,
        #     "variance_route_service_time" : 0
        # },
        # "solution_scope":0
        # }

        # solver_config_response = requests.post(
        #     url + "set_solver_config", params=data_params, json=solver_config
        # )
        # # print(f"SOLVER CONFIG ENDPOINT RESPONSE: {solver_config_response.json()}\n")

        # solve_parameters = {
        #     # Uncomment to disable/ignore constraints.

        #     # "ignore_capacities": True,
        #     # "ignore_vehicle_time_windows": True,
        #     # "ignore_vehicle_break_time_windows": True,
        #     # "ignore_task_time_windows": True,
        #     # "ignore_pickup_and_delivery": True,
        #     "return_status": False,
        #     "return_data_state": False,
        # }

        # solver_response = requests.get(
        #     url + "get_optimized_routes", params=solve_parameters, timeout=30
        # )
        
        # # Try multiple times.
        # if solver_response.status_code != 200:
        #     for _ in range(3):
        #         solver_response = requests.get(
        #             url + "get_optimized_routes", params=solve_parameters, timeout=30
        #         )
        #         if solver_response.status_code == 200:
        #             break
        # try:
        #     response = solver_response.json()["response"]["solver_response"]
        # except:
        #     print(solver_response)

        # for vehicle in response["vehicle_data"]:
        #     response["vehicle_data"][vehicle]["task_id"] = [floor(x/2) for x in response["vehicle_data"][vehicle]["task_id"]]
        # show_results(response)

        # for vehicle in response["vehicle_data"]:
        #     # response["vehicle_data"][vehicle]["task_id"] = [floor(x/2) for x in response["vehicle_data"][vehicle]["task_id"]]
        #     # print(response["vehicle_data"][vehicle]["task_id"])
            
        #     # pickup_locations = []
        #     # dropoff_locations = []

        #     cargo_dict = {}
        #     for cargo in state["active_cargo"]:
        #         cargo_dict[cargo.id] = [cargo.location, cargo.destination]

        #     # for plane in obs:
        #     #     for cargo_onboard in obs[plane]["cargo_onboard"]:
        #     #         if(obs[plane]["destination"] != 0):
        #     #             cargo_dict[cargo_onboard] = [obs[plane]["destination"], cargo_dict[cargo_onboard][1]]
        #     #         else:
        #     #             cargo_dict[cargo_onboard] = [obs[plane]["current_airport"], cargo_dict[cargo_onboard][1]]

        #     # print(cargo_dict)
        #     picked_up = []
        #     locations = []
        #     for task_id in response["vehicle_data"][vehicle]["task_id"]:
        #         if task_id not in picked_up:
        #             picked_up.append(task_id)
        #             # locations.append(cargo_dict[int(task_ids[task_id])][0])
        #             locations.append(cargo_dict[task_id][0])
        #         else:
        #             # locations.append(cargo_dict[int(task_ids[task_id])][1])
        #             locations.append(cargo_dict[task_id][1])

        #     # Create array from locations with number of same number in a row.
        #     offsets = []
        #     for i in range(len(locations)):
        #         if i == 0:
        #             offsets.append(1)
        #         elif locations[i] == locations[i-1]:
        #             offsets[-1] += 1
        #         else:
        #             offsets.append(1)

        #     grouped_tasks = []
        #     grouped_task_types = []
        #     for offset in offsets:
        #         grouped_tasks.append(response["vehicle_data"][vehicle]["task_id"][:offset])
        #         grouped_task_types.append(response["vehicle_data"][vehicle]["task_type"][:offset])
        #         response["vehicle_data"][vehicle]["task_id"] = response["vehicle_data"][vehicle]["task_id"][offset:]
        #         response["vehicle_data"][vehicle]["task_type"] = response["vehicle_data"][vehicle]["task_type"][offset:]

        #     response["vehicle_data"][vehicle]["task_id"] = grouped_tasks
        #     response["vehicle_data"][vehicle]["task_type"] = grouped_task_types
            
        # # print(f"SOLVER RESPONSE: {response}\n")

        # return response
    
    # def set_task_data(self, state):
        # task_locations = []#[0]
        # delivery_pairs = []
        # demand = []#[0]
        # task_ids = []
        # task_time_windows = []#[[0,100000]]
        # processing_time = []
        # penalties = []

        # cargo_dict = {}
        # for cargo in state["active_cargo"]:
        #     cargo_dict[cargo.id] = [cargo.location, cargo.destination]
        
        # for cargo in state["active_cargo"]:
        #     if(cargo.is_available):
        #         location = len(task_locations)
        #         task_locations.append(cargo_dict[cargo.id][0]-1)
        #         destination = len(task_locations)
        #         task_locations.append(cargo_dict[cargo.id][1]-1)
        #         delivery_pairs.append([location,destination])
        #         task_time_windows.append([cargo.earliest_pickup_time, cargo.soft_deadline])
        #         task_time_windows.append([cargo.earliest_pickup_time, cargo.soft_deadline])
        #         demand.append(cargo.weight)
        #         demand.append(-cargo.weight)
        #         task_ids.append(str(cargo.id))
        #         task_ids.append(str(cargo.id))
        #         processing_time.append(state["scenario_info"][0].processing_time)
        #         processing_time.append(state["scenario_info"][0].processing_time)
        #         penalties.append(100)
        #         penalties.append(100)
        
        # task_data = {
        #         "task_locations": task_locations,
        #         "demand": [demand],
        #         "task_time_windows": task_time_windows,
        #         "pickup_and_delivery_pairs": delivery_pairs,
        #         "task_ids": task_ids,
        #         # "service_time": processing_time,
        #         "penalties": penalties
        #     }
        
        # return task_data
        
    # def set_waypoint_data(self, state):
        # waypoints = {}
        # multidigraph = oh.get_multidigraph(state)
        # for plane_type in state["plane_types"]:
        #     waypoints[plane_type.id] = {"edges": [], "offsets": [], "weights": []}

        # nodes = list(dict(multidigraph.adj).keys())
        # print("  ", end =" ")
        # for node in nodes:
        #     print("{:>3}".format(node), end =" ")
        # print()

        # for node in nodes:
        #     print("{:>3}".format(node), end =" ")
        #     connections = list(dict(multidigraph.adj[node]).keys())
        #     for plane_type_id in list(waypoints.keys()):
        #         waypoints[plane_type_id]["offsets"].append(len(waypoints[plane_type_id]["edges"]))
        #     # size of weights is length of nodes.
        #     weights = [0]*len(nodes)

        #     for connection in connections:
        #         conn_by_plane_type = multidigraph.adj[node][connection]
        #         for plane_type_id in list(waypoints.keys()):
        #             if(plane_type_id in conn_by_plane_type):
        #                 weight = (multidigraph.adj[node][connection][plane_type_id]["time"]+state["scenario_info"][0].processing_time)
        #                 if not multidigraph.adj[node][connection][plane_type_id]["route_available"]:
        #                     weight += int(multidigraph.adj[node][connection][plane_type_id]["mal"])
        #                 weights[connection-1] = weight
        #                 waypoints[plane_type_id]["edges"].append(connection-1)
        #                 waypoints[plane_type_id]["weights"].append(weight)
            
        #     for weight in weights:
        #         # print with 5 characters
        #         # and right align it
        #         print("{:>3}".format(weight), end =" ")
        #     print()

        # for plane_type_id in list(waypoints.keys()):
        #     waypoints[plane_type_id]["offsets"].append(len(waypoints[plane_type_id]["edges"]))
        # return waypoints

    # def set_fleet_data(self, state):
        # fleet_data = {"capacities":[],"vehicle_locations":[],"vehicle_types": [], "vehicle_ids":[], "drop_return_trips" : []}
        # for agent in state["agents"]:
        #     if(state["agents"][agent]["state"] in [PlaneState.READY_FOR_TAKEOFF, PlaneState.WAITING]):
        #         fleet_data["capacities"].append(state["agents"][agent]["max_weight"])
        #         starting_position = state["agents"][agent]["current_airport"]-1
        #         if(state["agents"][agent]["destination"]!=0):
        #             starting_position = state["agents"][agent]["destination"]-1
        #         fleet_data["vehicle_locations"].append([starting_position,state["agents"][agent]["destination"]])
        #         fleet_data["vehicle_types"].append(state["agents"][agent]["plane_type"])
        #         fleet_data["vehicle_ids"].append(agent)
        #         fleet_data["drop_return_trips"].append(True)

        # # sort dictionary based on vehicle types
        # fleet_data["vehicle_types"], fleet_data["vehicle_locations"], fleet_data["capacities"], fleet_data["vehicle_ids"], fleet_data["drop_return_trips"] = zip(*sorted(zip(fleet_data["vehicle_types"], fleet_data["vehicle_locations"], fleet_data["capacities"], fleet_data["vehicle_ids"], fleet_data["drop_return_trips"])))
        # fleet_data["capacities"] = [fleet_data["capacities"]]
        # fleet_data["min_vehicles"] = 1

        # return fleet_data
    
    def collect_current_status(self, obs):
        state = self.get_state(obs)

        self.current_state["airport_info"] = {}
        for airport_id in range(1, len(state["route_map"][0].nodes)+1):
            self.current_state["airport_info"][airport_id] = {}
            self.current_state["airport_info"][airport_id]["cargo"] = []
            self.current_state["airport_info"][airport_id]["planes"] = []
        
        if("cargo_info" not in self.current_state):
            self.current_state["cargo_info"] = {}
        all_cargo_ids = list(self.current_state["cargo_info"].keys())
        for cargo in state["active_cargo"]:
            if(cargo.id in all_cargo_ids):
                all_cargo_ids.remove(cargo.id)
            self.current_state["cargo_info"][cargo.id] = {}
            self.current_state["cargo_info"][cargo.id]["location"] = cargo.location
            self.current_state["cargo_info"][cargo.id]["destination"] = cargo.destination
            self.current_state["cargo_info"][cargo.id]["pickup_time"] = cargo.earliest_pickup_time
            self.current_state["cargo_info"][cargo.id]["hard_deadline"] = cargo.hard_deadline
            self.current_state["cargo_info"][cargo.id]["soft_deadline"] = cargo.soft_deadline
            self.current_state["cargo_info"][cargo.id]["weight"] = cargo.weight
            self.current_state["cargo_info"][cargo.id]["status"] = "waiting"
            self.current_state["cargo_info"][cargo.id]["carrier"] = ""

            if(cargo.location!=0):
                self.current_state["airport_info"][cargo.location]["cargo"].append(cargo.id)

        self.current_state["agent_info"] = {}
        for agent_id in state["agents"]:
            cargo_in_transit = []
            self.current_state["agent_info"][agent_id] = {}
            current_agent = state["agents"][agent_id]
            next_action = current_agent["next_action"]
            cargo_on_board = current_agent["cargo_onboard"]
            plane_type_id = current_agent["plane_type"]
            current_airport = current_agent["current_airport"]
            plane_state = current_agent["state"]
            cargo_to_load = next_action["cargo_to_load"]
            cargo_to_unload = next_action["cargo_to_unload"]
            destination = 0

            if(plane_state in [PlaneState.PROCESSING]):
                for cargo_id in cargo_to_unload:
                    cargo_on_board.remove(cargo_id)
                for cargo_id in cargo_to_load:
                    cargo_on_board.append(cargo_id)
                destination = next_action["destination"]
            elif(plane_state in [PlaneState.MOVING]):
                destination = current_agent["destination"]

            cargo_in_transit.extend(cargo_on_board)

            if(destination!=0):
                current_airport = destination

            self.current_state["agent_info"][agent_id]["start"] = current_airport
            self.current_state["agent_info"][agent_id]["end"] = destination
            self.current_state["agent_info"][agent_id]["cargo_in_transit"] = cargo_in_transit
            self.current_state["agent_info"][agent_id]["state"] = plane_state
            self.current_state["agent_info"][agent_id]["plane_type_id"] = plane_type_id
            self.current_state["airport_info"][current_airport]["planes"].append(agent_id)
            self.current_state["agent_info"][agent_id]["capacity"] = current_agent["max_weight"]
            self.current_state["agent_info"][agent_id]["agent_index"] = len(list(self.current_state["agent_info"].keys()))

            for cargo_id in cargo_in_transit:
                if(cargo_id in all_cargo_ids):
                    all_cargo_ids.remove(cargo_id)
                self.current_state["cargo_info"][cargo_id]["status"] = "transit"
                self.current_state["cargo_info"][cargo_id]["carrier"] = agent_id
                self.current_state["cargo_info"][cargo_id]["location"] = current_airport
                self.current_state["airport_info"][current_airport]["cargo"].append(cargo_id)

            for cargo_id in all_cargo_ids:
                self.current_state["cargo_info"][cargo_id]["status"] = "completed"
                self.current_state["cargo_info"][cargo_id]["location"] = self.current_state["cargo_info"][cargo_id]["destination"]

        self.current_state["cost_graphs"] = {}
        for plane_type_id in state["route_map"]:
            self.current_state["cost_graphs"][plane_type_id] = []

            for airport_id in range(1, len(state["route_map"][0].nodes)+1):
                self.current_state["cost_graphs"][plane_type_id].append([100]*len(state["route_map"][0].nodes))
                self.current_state["cost_graphs"][plane_type_id][airport_id-1][airport_id-1] = 0
            
            for airport_id in state["route_map"][plane_type_id]:
                for destination_id in state["route_map"][plane_type_id][airport_id]:
                    self.current_state["cost_graphs"][plane_type_id][airport_id-1][destination_id-1] = state["route_map"][plane_type_id][airport_id][destination_id]["time"]
                    self.current_state["cost_graphs"][plane_type_id][destination_id-1][airport_id-1] += state["scenario_info"][0].processing_time * (1+(len(self.current_state["airport_info"][airport_id]["planes"]) // self.get_state(obs)['scenario_info'][0][1]))

                    if(not state["route_map"][plane_type_id][airport_id][destination_id]["route_available"]):
                        self.current_state["cost_graphs"][plane_type_id][airport_id-1][destination_id-1] += int(state["route_map"][plane_type_id].adj[airport_id][destination_id]["mal"])
                    
        