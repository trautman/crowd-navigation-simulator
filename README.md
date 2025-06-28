
# Crowd Navigation Simulator

A Python-based simulator for evaluating social navigation policies in crowd environments using the ORCA model. This tool supports batch simulations, configurable environments, and a robust analysis suite.

## Repository

**GitHub:** [https://github.com/trautman/crowd-navigation-simulator](https://github.com/trautman/crowd-navigation-simulator)

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/trautman/crowd-navigation-simulator.git
cd crowd-navigation-simulator
```

### 2. Run a simulation

```bash
python simulator.py \
    --env-config env-configs/boardwalk.yaml \
    --sim-config trials/<site>/<scenario>/RUN/simulator_config.yaml \
    --write_data_to trials/<site>/<scenario>/RUN/data/ \
    --gui
```

Replace `<site>` and `<scenario>` accordingly.

---

## 🛠️ Simulator Configuration

### Create a Simulation Run

1. In the `trials/` directory, create a folder for your desired setup:
    ```bash
    trials/<site>/<scenario>/RUN/
    ```

2. Place your `simulator_config.yaml` inside this `RUN` directory.

3. Create a `data/` folder inside the same `RUN` directory. This is where the simulator outputs metrics and other data.

    Example structure:
    ```
    trials/
      └── dwb/
          └── arcade/
              └── baseline_config/
                  ├── RUN/
                  │   └── simulator_config.yaml
                  └── data/
    ```

---

## 📊 Analysis Instructions

### Giant Summary Plot

Generate a comprehensive set of metrics binned and plotted with regression:

```bash
cd analysis
python analyze_bags.py \
    --data-dir ../trials/dwb/arcade/baseline_config/data/ \
    --binned --all --giant_plot
```

### Plot Individual Metrics

- All binned plots:
  ```bash
  python analyze_bags.py --data-dir ../trials/dwb/arcade/baseline_config/data/ --binned --all
  ```

- Just efficiency (binned):
  ```bash
  python analyze_bags.py --data-dir ../trials/dwb/arcade/baseline_config/data/ --binned --efficiency
  ```

- Scatter version of efficiency:
  ```bash
  python analyze_bags.py --data-dir ../trials/dwb/arcade/baseline_config/data/ --scatter --efficiency
  ```

### YAML Metadata Generation

To create yaml summaries for sim-real-sim2real processing:

```bash
python analyze_bags.py \
    --data-dir ../trials/dwb/arcade/baseline_config/data/ \
    --write_to_yaml \
    --site Arcade \
    --state ORCA \
    --baseline DWB
```

---

## 🧹 Clean Trials

Remove trials with excessive path lengths:

```bash
python clean_by_path_length.py \
    --data-dir ../trials/brne/spawn_rates_1.5-6_spawners_7_old_brne_cost_weights_YES_MASK_with_boundary_correction/data/ \
    --thresh 12
```


---

## 🧾 Notes

- The simulator reads parameters from `simulator_config.yaml`.
- Data is written into the `data/` directory inside each RUN folder.
- GUI mode (`--gui`) is optional but useful for visualization.

---

## 📁 Directory Summary

```
crowd-navigation-simulator/
├── simulator.py
├── env-configs/
├── trials/
│   └── <site>/<scenario>/RUN/simulator_config.yaml
│   └── <site>/<scenario>/data/
└── analysis/
    ├── analyze_bags.py
    └── clean_by_path_length.py
```

---

