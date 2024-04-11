# HPO-RL-Bench


### Download the data

Get the data from [HERE](https://drive.google.com/file/d/1AW5_6xGGiklteZgyyDBxSsf6kOLuFPkO/view?usp=share_link), download this repo and put it at the level of this repository folder.

### Install Requirements

`conda create -n hpo_rl_bench python=3.9`

`conda activate hpo_rl_bench`

`conda install swig`

`conda install -r requirements.txt`


**Note:** For *pyrfr*,  Microsoft Visual C++ 14.0 or greater is required. Get it with [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). 

### Load and query the benchmark

```python
from benchmark_handler import BenchmarkHandler

benchmark = BenchmarkHandler(environment="Enduro-v0", seed=0,
                             search_space="PPO", set="static")

# querying static configuration
configuration_to_query = {"lr": -6, "gamma": 0.8, "clip": 0.2}
queried_data = benchmark.get_metrics(configuration_to_query, budget=50)

# querying dynamic configuration
benchmark.set = "dynamic"
configuration_to_query = {"lr": [-3, -4],
                          "gamma": [0.98, 0.99],
                          "clip": [0.2, 0.2]}
queried_data = benchmark.get_metrics(configuration_to_query, budget=50)

```

### Further usage

For an insightful usage description please check the file `benchmark-usages-examples.ipynb`

### Reproducing plots from the paper
Run *plot_static_ppo.py* to generate Figure 4.

Run *plot_dynamic.py* to generate Figure 5a.

Run *plot_extended.py* to generate Figure 5b.

Run *cd_diagram.py* to generate Figure 6a.

To generate Figure 6b, change `ALGORITHM="A2C"` in line 384 of *cd_diagram.py* and run it.

To generate Figure 2 and 3, we use *benchmark_EDA.py*.