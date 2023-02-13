# gator-lift
The repository containing the submission for the Airlift Challenge by the team at the University of Florida

## Clone Repo
```
git clone https://github.com/UF-Airlift-Challenge-2023/gator-lift.git
cd gator-lift
git switch graph-formatting
git submodule update --init --recursive
```

### Activate Conda
```
conda activate airlift-solution
```

## Install airlift python module
```
pip install -e airlift/.
```
### From Spider
```
%pip install -e airlift/.
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
### From Spider
```
%run airlift-starter-kit/eval_solution.py --scenario-folder airlift-starter-kit/scenarios
```

## Run current solution code
```
python airlifttest.py
```

### From Spider
```
%run airlifttest.py
```

# Run CuOpt on HiperGator
This can be used to run a python shell for itneracting with cuopt.
```
apptainer run --nv /apps/nvidia/containers/cuOpt/22.12 python
```

This next command is used to start a Jupyter Notebook that contains the cuopt resources.
```
apptainer run --nv -B /blue/{lab-dir}/{username}/home/cuopt_user/.local:/home/cuopt_user/.local /apps/nvidia/containers/cuOpt/22.12 jupyter-notebook --NotebookApp.ip='0.0.0.0' --NotebookApp.port=23312 --NotebookApp.token=6f11e3c7b0d5e23f07df7257d73861329a5d216fe85cd9ab
```
After running this command you must open a ssh tunnel.
```
ssh -NL 23312:{unique-identifier}.ufhpc:23312 {username}@hpg.rc.ufl.edu
```

The unique identifier should be specified after running the jupyter notebook in a terminal print out immeditaly after `Jupyter Notebook 6.5.2 is running at:`. By running the ssh tunnel you will be able to access the jupyter notebook via `http://127.0.0.1:23312/?token=6f11e3c7b0d5e23f07df7257d73861329a5d216fe85cd9ab` in your web browser. (Note you will ahve to perform these steps on your computer and rather than using my username you will need to use yours.)

Or alternatively you can the same URL to VSCode as a Jupyter notebook server. That way when using VSCode and running notebooks it will run it on the remote server rather than your local device. This allows you to store all your files on your local directory and simply run them in the environment on HiperGator.
