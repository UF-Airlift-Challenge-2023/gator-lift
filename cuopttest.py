import requests
import pandas as pd

ip = "127.0.0.1"
port = "5000"
url = "http://" + ip + ":" + port + "/cuopt/"

data_params = {"return_data_state": False}


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
    
waypoint_graph = {
    "waypoint_graph":{
        "0":
        {
            "offsets": [0,       3,    5,           9,    11,   13,   15,   17, 18, 19, 20, 21], # noqa
            "edges":   [1, 2, 9, 0, 7, 0, 3, 4, 10, 2, 4, 2, 5, 6, 9, 5, 8, 1,  6,  0,  5], # noqa
            "weights": [1, 1, 2, 1, 2, 1, 1, 1,  3, 2, 3, 2, 1, 2, 1, 3, 4, 2,  3,  1,  1]  # noqa
        }
    }
    }

matrix_response = requests.post(
    url + "set_cost_waypoint_graph", params=data_params, json=waypoint_graph
)
print(f"\nWAYPOINT GRAPH ENDPOINT RESPONSE: {matrix_response.json()}\n")

fleet_data = {
    "vehicle_locations": [[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]],
    "capacities": [[10, 12, 15, 8, 10]],
    "vehicle_time_windows": [[0, 80], [1, 40], [3, 30], [5, 80], [20, 100]],
    "vehicle_types": [0, 1, 0, 0, 0],
}

fleet_response = requests.post(
    url + "set_fleet_data", params=data_params, json=fleet_data
)
print(f"FLEET ENDPOINT RESPONSE: {fleet_response.json()}\n")

task_data = {
    "task_locations": [0, 1, 3, 4, 6, 8],
    "demand": [[0, 3, 4, 4, 3, 2]],
    "task_time_windows": [
        [0, 1000],
        [3, 20],
        [5, 30],
        [1, 20],
        [4, 40],
        [0, 30],
    ],
    "service_times": [0, 3, 1, 8, 4, 0],
}

task_response = requests.post(
    url + "set_task_data", params=data_params, json=task_data
)

print(f"TASK ENDPOINT RESPONSE: {task_response.json()}\n")

solver_config = {"time_limit": 0.01, "number_of_climbers": 128}

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
    url + "get_optimized_routes", params=solve_parameters
)
print(f"SOLVER RESPONSE: {solver_response.json()}\n")
show_results(solver_response.json()["response"]["solver_response"])