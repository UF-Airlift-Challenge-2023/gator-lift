# gator-lift
The repository containing the submission for the Airlift Challenge by the team at the University of Florida

## Clone Repo
```
git clone https://github.com/UF-Airlift-Challenge-2023/gator-lift.git
cd gator-lift
git switch graph-formatting
git submodule update --init --recursive
```

## Install airlift python module
```
cd airlift
pip install -e .
```

## Used to download the test scenarios
```
curl https://airliftchallenge.com/scenarios/airlift_test_scenarios.zip -O -J -L
tar -xf airlift_test_scenarios.zip -C airlift-starter-kit/scenarios
```

## Used to run eval_solution.py and confirm install
```
python airlift-starter-kit/eval_solution.py --scenario-folder airlift-starter-kit/scenarios
```
