from matplotlib import tight_layout
from benchmark_handler import BenchmarkHandler
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import feature_selection as f


sns.set() # Setting seaborn as default style even if use only matplotlib
def get_aggregated_results (benchmark, space_name="PPO", budget= 99, metric = "returns_eval", smoke_test = False):

    search_space = benchmark.get_search_space(space_name)
    hps_names = list(search_space.keys())
    environments = benchmark.get_environments()

    if smoke_test:
        environments = environments[:3]

    ranks = []
    all_results = []
    failed_confs = []
    for seed in [0,1,2]:
        temp_seed_ranks = []
        for environment in environments:
            results_history = []
            
            if space_name == "SAC" and environment=="Ant-v2":
                print("SAC")
            try: 
                benchmark.set_env_space_seed(search_space=space_name, environment=environment, seed=seed)
                temp_failed_confs = []
                for hps in itertools.product(*tuple(list(search_space.values()))):
                    try:
                        configuration = dict(zip(hps_names, hps))
                        query = benchmark.get_metrics(configuration, budget=budget)[metric]
                        configuration["response"] = query[-1]
                        configuration[metric] = metric
                        configuration["environment"] = environment
                        configuration["conf_id"] =  str(hps)
                        results_history.append(configuration)
                    except Exception as e:
                        print(e)
                        temp_failed_confs.append(hps)
                        print(f"Problem at config: {configuration}")

                results_df = pd.DataFrame(results_history)
                rank = results_df["response"].rank(ascending=False).values.tolist()
                temp_seed_ranks.append(rank)
                all_results.append(results_df)
                failed_confs.extend(temp_failed_confs)
                all_configurations = results_df["conf_id"]
            except Exception as e:
                #print("env:", environment, " space:", space_name)
                print(e)
                pass

        ranks.append(temp_seed_ranks)
    #failed_confs = list(set(failed_confs))
    #all_configurations = list(itertools.product(*tuple(search_space.values())))
    #all_configurations = [str(x) for x in all_configurations if x not in failed_confs ]
    aggregated_results = pd.DataFrame()
    for rank in ranks:
        temp_df = pd.DataFrame(rank)
        temp_df = temp_df.T.dropna().T
        temp_df.columns = all_configurations
        aggregated_results = pd.concat((aggregated_results, temp_df), axis=0)

    features_rank = []
    for result in all_results:
        temp_rank=pd.DataFrame(f.f_regression(result[hps_names], result["response"])[0]).rank(ascending=False).values.reshape(-1).tolist()
        #temp_rank=pd.DataFrame(f.f_regression(result[hps_names], result["response"])[0]).values.reshape(-1).tolist()
        features_rank.append(temp_rank)

    return aggregated_results, features_rank, all_configurations, hps_names


def plot_catplot_on_axis(benchmark, space_name, smoke_test = False):

    aggregated_results, features_rank, all_configurations, hps = get_aggregated_results(benchmark, space_name, smoke_test = smoke_test)

    mean_aggregated_results = aggregated_results.mean(axis=0)
    ix_sort = mean_aggregated_results.argsort()
    best = ix_sort[:n_configurations_to_plot].tolist()
    worst = ix_sort[-n_configurations_to_plot:].tolist()
    selected_configurations = []
    for config_id in best+worst:
        selected_configurations.append(all_configurations[config_id])

    subset_results = pd.DataFrame(aggregated_results.iloc[:, best+worst])
    subset_results.columns = selected_configurations

    configuration_structure = str(tuple (benchmark.get_search_space(space_name).keys()))
    return subset_results, features_rank, configuration_structure, hps 
    

data_path = ""
benchmark = BenchmarkHandler(data_path=data_path)


n_configurations_to_plot = 4
linewidth = 2
smoke_test = False
fontsize=70
boxprops = dict(linestyle='--', linewidth=linewidth)
flierprops = dict(marker='o', markerfacecolor='green', markersize=12,
                  markeredgecolor='none')
medianprops = dict(linestyle='-', linewidth=linewidth*3, color='firebrick')
meanpointprops = dict(marker='D', markeredgecolor='black', markersize=12,  markerfacecolor='firebrick')
meanlineprops = dict(linestyle='--', linewidth=linewidth, color='purple')
algorithms = ["PPO","A2C", "DDPG", "SAC", "TD3"]
n_algorithms = len(algorithms)
fig1, axis1 = plt.subplots(1,n_algorithms,figsize=(15*n_algorithms,15))
fig2, axis2 = plt.subplots(1,n_algorithms,figsize=(15*n_algorithms,15))

plt.rcParams.update({
    "font.family": "serif",

})
   
for i, space_name in enumerate(algorithms):
    subset_results, features_rank, configuration_structure, hps = plot_catplot_on_axis(benchmark, space_name, smoke_test)
    selected_configurations = subset_results.columns

    bplot1 = axis1[i].boxplot(subset_results, patch_artist=True, boxprops=boxprops,
                                                            flierprops=flierprops,
                                                            meanprops=meanpointprops,
                                                            medianprops=medianprops)

    features_rank  = np.array(features_rank)
    features_rank[np.isnan(features_rank)] = 1
    bplot2 = axis2[i].boxplot(features_rank, patch_artist=True, boxprops=boxprops,
                                                            flierprops=flierprops,
                                                           # meanpointprops=meanpointprops,
                                                            meanprops=meanpointprops,
                                                            medianprops=medianprops,
                                                             showmeans=True)
    
    #axis2[i].violinplot(np.array(features_rank))

    n  = 8
    cm = plt.cm.get_cmap("Paired")
    colors = [cm(val/n) for val in range(n)]

    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    axis1[i].set_xticklabels(selected_configurations, rotation=90, fontsize=fontsize)
    axis1[i].yaxis.set_tick_params(labelsize=fontsize)
    axis1[i].set_title(space_name, fontsize=fontsize)
    axis1[i].set_xlabel(configuration_structure, fontsize=fontsize)


    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)

    axis2[i].set_xticklabels(hps, rotation=0, fontsize=fontsize)
    axis2[i].yaxis.set_tick_params(labelsize=fontsize)
    axis2[i].set_title(space_name, fontsize=fontsize)
    
    if i==0:
        axis1[i].set_ylabel("Rank of Reward", fontsize=fontsize)
    axis1[i].xaxis.set_label_coords(.5, -.8)

    if i==0:
        axis2[i].set_ylabel("Rank of Importance", fontsize=fontsize)

    if i==2:
        axis2[i].set_xlabel("Hyperparameters", fontsize=fontsize)

    axis2[i].xaxis.set_label_coords(.5, -.2)
#plt.show()    

plt.tight_layout()
fig1.savefig("plots/average_rank_per_algorithm.pdf", bbox_inches="tight")
fig2.savefig("plots/feature_importance_per_algorithm.pdf", bbox_inches="tight")