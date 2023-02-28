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
        # clear_request.json()
        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)
        self.new_change = True
        self.new_state = self.get_state(obs)
        self.down_routes = []
        self.prev_occupied = []

    def policies(self, obs, dones):

        occupied_airports = list()
        for plane in obs:
            current_plane = obs[plane]
            if current_plane['state'] != 2: # not 2 = at an airport somewhere
                occupied_airports.append(current_plane['current_airport'])
            else: # 2 = in flight to destination, therefore not in any airport
                occupied_airports.append(current_plane['destination'])

        '''
        2. iterate through list, perform calculation
        --> if numPlanes(airport) >= working_capacity for all airports
        ----> state[scenarioInfo][0][working_capacity]
        ----> then edge = P*(1 + numPlanes(airport)//processingTime)
        '''

        # determine which airports are at capacity
            # obtain processing time and working capacity
            # index 0 of scenario is processing time, index 1 is working capacity
        processing_time = self.get_state(obs)['scenario_info'][0][0]
        working_capacity = self.get_state(obs)['scenario_info'][0][1]
        state = self.get_state(obs)
        self.multidigraph = oh.get_multidigraph(state)

        for airport_number in occupied_airports:
            # number of planes per airport
            numPlanes = occupied_airports.count(airport_number)
            if numPlanes > working_capacity:
                if airport_number not in self.prev_occupied:
                    self.prev_occupied.append(airport_number)
                    # self.new_change = True
                difference = numPlanes - working_capacity
                extra_wait_time = processing_time * (1 + (numPlanes // working_capacity))
                connections = list(self.multidigraph.adj[airport_number].keys())
                for connection in connections:
                    for plane_type in self.multidigraph.adj[connection][airport_number]:
                        self.multidigraph.adj[connection][airport_number][plane_type]["time"] += extra_wait_time
            else:
                if airport_number in self.prev_occupied:
                    self.prev_occupied.remove(airport_number)
                    # self.new_change = True
        
        if (len(self.get_state(obs)["event_new_cargo"]) != 0):
            # self.new_change = True
            print("New cargo detected!")
        
        if(self.new_change):
            self.solver_response = self.process_state(obs)
            self.new_state = self.get_state(obs)
            self.new_change = False

        
        state = self.get_state(obs)
        for type in state["route_map"]:
            for node in list(dict(state["route_map"][type].adj).keys()):
                for connection in list(dict(state["route_map"][type].adj[node]).keys()):
                    if(state["route_map"][type].adj[node][connection]["mal"]!=0):
                        if([node, connection] in self.down_routes):
                            print("DOWN ROUTE: " + str([node, connection]))
                        # print(node, connection, state["route_map"][type].adj[node][connection]["mal"])
        
        
        my_action = self.process_response(obs)
        
        # Use the acion helper to generate an action

        # return None
        random_action = self._action_helper.sample_valid_actions(obs)
        # 'a_0' : {'process': 0, 'cargo_to_load': [], 'cargo_to_unload': [], 'destination': 0}
        return random_action

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

        self.task_data = set_task_data(state)
        
        self.cargo_assignments = {a: None for a in self.agents}
        self.path = {a: None for a in self.agents}
        self.whole_path = {a: None for a in self.agents}
        self.plane_type_waypoints = {}

        self.fleet_data = set_fleet_data(state)

        self.waypoints = set_waypoint_data(state)
        
        self.waypoint_graph = {
            "waypoint_graph":self.waypoints
        }

        self.fleet_data = self.fleet_data

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

            # for plane in obs:
            #     for cargo_onboard in obs[plane]["cargo_onboard"]:
            #         if(obs[plane]["destination"] != 0):
            #             cargo_dict[cargo_onboard] = [obs[plane]["destination"], cargo_dict[cargo_onboard][1]]
            #         else:
            #             cargo_dict[cargo_onboard] = [obs[plane]["current_airport"], cargo_dict[cargo_onboard][1]]

            # print(cargo_dict)
            picked_up = []
            locations = []
            for task_id in response["vehicle_data"][vehicle]["task_id"]:
                if task_id not in picked_up:
                    picked_up.append(task_id)
                    # locations.append(cargo_dict[int(task_ids[task_id])][0])
                    locations.append(cargo_dict[task_id][0])
                else:
                    # locations.append(cargo_dict[int(task_ids[task_id])][1])
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
    
def set_task_data(self, state):
    task_locations = []#[0]
    delivery_pairs = []
    demand = []#[0]
    task_ids = []
    task_time_windows = []#[[0,100000]]
    processing_time = []
    penalties = []

    cargo_dict = {}
    for cargo in state["active_cargo"]:
        cargo_dict[cargo.id] = [cargo.location, cargo.destination]
    
    for cargo in state["active_cargo"]:
        if(cargo.is_available):
            location = len(task_locations)
            task_locations.append(cargo_dict[cargo.id][0]-1)
            destination = len(task_locations)
            task_locations.append(cargo_dict[cargo.id][1]-1)
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
    
    task_data = {
            "task_locations": task_locations,
            "demand": [demand],
            "task_time_windows": task_time_windows,
            "pickup_and_delivery_pairs": delivery_pairs,
            "task_ids": task_ids,
            # "service_time": processing_time,
            "penalties": penalties
        }
    
    return task_data
    
def set_waypoint_data(self, state):
    waypoints = {}
    multidigraph = oh.get_multidigraph(state)
    for plane_type in state["plane_types"]:
        waypoints[plane_type.id] = {"edges": [], "offsets": [], "weights": []}

    nodes = list(dict(multidigraph.adj).keys())
    print("  ", end =" ")
    for node in nodes:
        print("{:>3}".format(node), end =" ")
    print()

    for node in nodes:
        print("{:>3}".format(node), end =" ")
        connections = list(dict(multidigraph.adj[node]).keys())
        for plane_type_id in list(waypoints.keys()):
            waypoints[plane_type_id]["offsets"].append(len(waypoints[plane_type_id]["edges"]))
        # size of weights is length of nodes.
        weights = [0]*len(nodes)

        for connection in connections:
            conn_by_plane_type = multidigraph.adj[node][connection]
            for plane_type_id in list(waypoints.keys()):
                if(plane_type_id in conn_by_plane_type):
                    weight = (multidigraph.adj[node][connection][plane_type_id]["time"]+state["scenario_info"][0].processing_time)
                    if not multidigraph.adj[node][connection][plane_type_id]["route_available"]:
                        weight += int(multidigraph.adj[node][connection][plane_type_id]["mal"])
                    weights[connection-1] = weight
                    waypoints[plane_type_id]["edges"].append(connection-1)
                    waypoints[plane_type_id]["weights"].append(weight)
        
        for weight in weights:
            # print with 5 characters
            # and right align it
            print("{:>3}".format(weight), end =" ")
        print()

    for plane_type_id in list(waypoints.keys()):
        waypoints[plane_type_id]["offsets"].append(len(waypoints[plane_type_id]["edges"]))
    return waypoints

def set_fleet_data(self, state):
    fleet_data = {"capacities":[],"vehicle_locations":[],"vehicle_types": [], "vehicle_ids":[], "drop_return_trips" : []}
    for agent in state["agents"]:
        if(state["agents"][agent]["state"] in [PlaneState.READY_FOR_TAKEOFF, PlaneState.WAITING]):
            fleet_data["capacities"].append(state["agents"][agent]["max_weight"])
            starting_position = state["agents"][agent]["current_airport"]-1
            if(state["agents"][agent]["destination"]!=0):
                starting_position = state["agents"][agent]["destination"]-1
            fleet_data["vehicle_locations"].append([starting_position,state["agents"][agent]["destination"]])
            fleet_data["vehicle_types"].append(state["agents"][agent]["plane_type"])
            fleet_data["vehicle_ids"].append(agent)
            fleet_data["drop_return_trips"].append(True)

    # sort dictionary based on vehicle types
    fleet_data["vehicle_types"], fleet_data["vehicle_locations"], fleet_data["capacities"], fleet_data["vehicle_ids"], fleet_data["drop_return_trips"] = zip(*sorted(zip(fleet_data["vehicle_types"], fleet_data["vehicle_locations"], fleet_data["capacities"], fleet_data["vehicle_ids"], fleet_data["drop_return_trips"])))
    fleet_data["capacities"] = [fleet_data["capacities"]]
    fleet_data["min_vehicles"] = 1

    return fleet_data

# def collect_current_status(self, obs):
#     self.state = self.get_state(obs)
#     transit_times = previous_data["transit_time_matrices"][0]
#     task_locations = previous_data["task_locations"]
#     vehicle_start_locations = [val[0] for val in previous_data["vehicle_locations"]]
#     vehicle_return_locations = [val[1] for val in previous_data["vehicle_locations"]]
    
#     # Create a mapping between pickup and delivery
#     pickup_of_delivery = {
#         previous_data["delivery_indices"][i]: previous_data["pickup_indices"][i]
#         for i in range(len(previous_data["pickup_indices"]))
#     }
    
#     # Update vehicle earliest if needs to be changed to current time
#     vehicle_earliest = [max(earliest, reroute_from_time) for earliest in previous_data["vehicle_earliest"]]
    
#     # Collect completed and partial set of tasks, so we can add partialy completed tasks back
#     completed_tasks = []
#     picked_up_but_not_delivered = {}
#     picked_up_task_to_vehicle = {}
    
#     for veh_id, veh_data in optimized_route_data["vehicle_data"].items():
#         route_len = len(veh_data['route'])
#         task_len = len(veh_data["task_id"])
#         vehicle_id = int(veh_id)
    
#         # In this case, all the tasks are already completed, or waiting on last task service time
#         if veh_data['arrival_stamp'][-1] <= reroute_from_time:
#             intra_task_id = task_len
#         else:
#             try:
#                 # Look for a task that is yet to be completed
#                 intra_task_id, time = next(
#                     (i, el)
#                     for i, el in enumerate(veh_data['arrival_stamp'])
#                     if el > reroute_from_time
#                 )
#             except StopIteration:
#                 # In case none of the tasks are completed
#                 intra_task_id = 0
#                 time = max(vehicle_earliest[vehicle_id], reroute_from_time)

#         # All the tasks are completed and vehicle is on the way to return location or already reached

#         picked_up_but_not_delivered[vehicle_id] = []
            
#         # There are tasks that are still pending
#         if intra_task_id < task_len:
#             last_task = veh_data["task_id"][intra_task_id]
            
#             # Update vehicle start location
#             vehicle_start_locations[int(vehicle_id)] = task_locations[last_task]
            
#             # Update vehicle earliest
#             vehicle_earliest[int(vehicle_id)] = min(
#                 max(time, reroute_from_time), previous_data["vehicle_latest"][vehicle_id]
#             )
            
#             for j in range(0, intra_task_id):
#                 task = veh_data["task_id"][j]
#                 if task in previous_data["pickup_indices"]:
#                     picked_up_but_not_delivered[vehicle_id].append(task)
#                     picked_up_task_to_vehicle[task] = vehicle_id
#                 else:
#                     # Moves any delivered pick-up tasks to completed.
#                     corresponding_pickup = pickup_of_delivery[task]
#                     picked_up_but_not_delivered[vehicle_id].remove(
#                             corresponding_pickup
#                     )
#                     completed_tasks.append(corresponding_pickup)
#                     completed_tasks.append(task)
#                     picked_up_task_to_vehicle.pop(corresponding_pickup)
#         else:
#             completed_tasks.extend(veh_data["task_id"])
#             # In this case vehicle is at last location about to finish the task,
#             # so vehicle start location would last task location and accordingly the earliest vehicle time as well
#             if (veh_data['arrival_stamp'][-1] == reroute_from_time) and (veh_data['arrival_stamp'][-1]+previous_data["task_service_time"][veh_data["task_id"][-1]] >= reroute_from_time):
#                 vehicle_start_locations[vehicle_id] = task_locations[veh_data["task_id"][-1]]
#                 vehicle_earliest[vehicle_id] = veh_data['arrival_stamp'][-1] + previous_data["task_service_time"][veh_data["task_id"][-1]] 
#             else:
#                 # In this case vehicle completed last task and may be enroute to vehicle return location or might have reached.
#                 end_time = (
#                     veh_data['arrival_stamp'][-1] + previous_data["task_service_time"][veh_data["task_id"][-1]] + transit_times[task_locations[veh_data["task_id"][-1]]][vehicle_return_locations[vehicle_id]]
#                 )
#                 time = max(end_time, reroute_from_time)
#                 print("For vehicle ID updating", vehicle_id)
#                 vehicle_start_locations[vehicle_id] = vehicle_return_locations[vehicle_id]
#                 vehicle_earliest[vehicle_id] = min(time, previous_data["vehicle_earliest"][vehicle_id])
                
#     return (
#         vehicle_earliest, vehicle_start_locations,
#         vehicle_return_locations, completed_tasks,
#         picked_up_but_not_delivered, picked_up_task_to_vehicle)
