# HPO-RL-Bench


### Download the data

Get the data from [HERE](https://rewind.tf.uni-freiburg.de/index.php/s/R9FwPznPecJRqip), download this repo and put it at the level of this repository folder.

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

For an insightful usage description please check the file `benchmark-usages-examples.ipynb`



