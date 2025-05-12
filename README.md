# ORCA‐Sim: BRNE vs DWA in ORCA Pedestrian Crowds
In the root folder is /trials.  in /trials, we have brne/ and dwa/.  In both of thes, we store simulator configurations---for example, baseline_config/.  In this folder we store the simulator_config.yaml file, and create a data/ folder which is where the data gets written to. we run the simulator inside of data/ folder.  

~/Desktop/simulation/home-grown-simulator/trials/brne/baseline_config/data$ python ../../../../simulator.py --env-config ../../../../boardwalk.yaml --sim-config ../simulator_config_brne.yaml 
