from airlift.evaluators.utils import doeval
from mysolution import MySolution

def run_evaluation(scenario_folder):
    doeval(scenario_folder, MySolution())

if __name__ == "__main__":
    run_evaluation("airlift-starter-kit/scenarios")