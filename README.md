# HPO-RL-Bench


### Download the data

Get the data from [HERE](https://drive.google.com/file/d/1AW5_6xGGiklteZgyyDBxSsf6kOLuFPkO/view?usp=share_link), download this repo and put it at the level of this repository folder.

### Install Requirements

`conda create -n hpo_rl_bench python=3.9`

`conda activate hpo_rl_bench`

`conda install -r requirements.txt`


**Note:** You might need to install additional libraries for C++. 

### Load and query the benchmark

```python
from benchmark_handler import BenchmarkHandler

benchmark = BenchmarkHandler(data_path = "/data_arl_bench",
                             environment = "Pendulum-v0", seed = 0,
                             search_space = "PPO", static = True)

#querying static configuration
configuration_to_query = {"lr":-6, "gamma": 0.8, "clip": 0.2}
queried_data = benchmark.get_metrics(configuration_to_query, budget=50)

#querying dynamic configuration
benchmark.static = False
configuration_to_query = {"lr":[-3,-4], 
                          "gamma": [0.8,0.99], 
                          "clip": [0.2, 0.2]}
queried_data = benchmark.get_metrics(configuration_to_query, budget=50)

```

### Further usage

Try directly the benchmark by executing [this example](https://colab.research.google.com/drive/1mDKkXVP7Tf_IzZ6AYVrkFrZsoJZ93TgZ?usp=sharing) on our Jupyter Notebook.
For an insightful usage description please check the file `benchmark-usages-examples.ipynb`



