import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy import interpolate
from matplotlib.cm import get_cmap
import pandas as pd
from scipy import stats
from cd_diagram import draw_cd_diagram as draw
from benchmark_handler import BenchmarkHandler


def get_filename (optimizer, search_space, environment, seed, results_path="results", optimizer_path = None):

    if optimizer_path is None:
        optimizer_path = optimizer
    pb_optimizers = ["PBT", "PB2"]
    if optimizer in pb_optimizers and optimizer_path in pb_optimizers:
        filename =  f"{optimizer}_seed{seed}_{environment}_{search_space}_full.json"
    else:
        filename =  f"{optimizer}_seed{seed}_{environment}_{search_space}.json"

    return os.path.join(results_path, optimizer_path, filename)

def collect_results(optimizer, search_space, environments, seeds, omit=[], optimizer_path = None):

    collected_results = {"incumbents":[], "wallclock_time_eval":[], "timesteps_eval":[]}
    to_omit = []
    for env in environments:
        for seed in seeds:

            if (env,seed) not in omit:
                filename =  get_filename(optimizer, search_space, env, seed, optimizer_path=optimizer_path)

                try:

                    with open(filename) as f:
                        data = json.load(f)

                    for field in collected_results.keys():
                        data_field = data[field][:100]
                        data_field_len = len(data_field)
                        if  data_field_len< 100:
                            data_field+= [np.nan]*(100-data_field_len)
                        collected_results[field].append(data_field)

                except Exception as e:
                    print(e)
                    to_omit.append((env,seed))

    return collected_results, to_omit

def discretized_grid (incumbents, budget, budget_range, points = 500):


    
    budget_grid = np.linspace(budget_range[0], budget_range[1], num=points)

    incumbents_on_grid = []
    for i in range(incumbents.shape[0]):
        upper_incumbent = incumbents[i][-1]
        f = interpolate.interp1d(budget[i], incumbents[i], bounds_error=False, fill_value=(np.nan, upper_incumbent), kind="nearest")
        incumbents_on_grid.append(f(budget_grid).tolist())

    return np.array(incumbents_on_grid), budget_grid



def draw_cd_diagram(self,  bo_iter=50, name="Rank", path=None):

    path = path if path is not None else self.path
    df = pd.DataFrame(np.array(self.all_ranks )[:,:,bo_iter].T.tolist()).T
    df.columns = self.experiments
    df = df.stack().reset_index()
    df.columns = ["dataset_name", "classifier_name", "accuracy"]
    df.accuracy = -df.accuracy
    draw(df, path_name= path+name+".png", title=name)


def plots_on_axis( axis, results_list, x_list, title="",   draw_std=False, 
                                                        xscale = "linear",
                                                        yscale = "linear", 
                                                        legends=None,
                                                        xticks = None,
                                                        xticks_labels =None,
                                                        fontsize = 20,
                                                        linewidth = 3,
                                                        xlim = None,
                                                        xlabel = "No. of Steps" ,
                                                        ylabel="Average Rank", 
                                                        colors = None):


    results = np.array(results_list)
    x = np.array(x_list)

    #regret = np.array(regret_list)[:,:,-96:]
    sample_size, n_experiments, n_bo_iters = results.shape

    mean = np.nanmean(results,axis=0)
    #regret_mean = np.nanmean(regret,axis=0)
    std = np.nanstd(results,axis=0)
    #regret_std = np.nanstd(regret,axis=0)
    ci_factor = 1.96/np.sqrt(sample_size)

    if xticks==None:
        xticks = x[np.linspace(0, len(x)-1, 5).astype(int)]
        xticks_labels = xticks


    if colors == None:
        cmap = get_cmap(name='tab10')
        n_colors = len(results_list[0])
        colors = cmap(np.linspace(0, 1, n_colors ))
        

    for k in range(mean.shape[0]):
        y = mean[k,:]
        
        if legends != None:
            label = legends[k]
        else:
            label = None

        axis.plot(x, y,  linewidth=linewidth, color = colors[k], label=label)

        if draw_std:
            y_std = std[k,:]*ci_factor*0.5
            ci1 = y-y_std
            ci2 = y+y_std
            axis.fill_between( x, ci1, ci2, alpha=.1)

    if yscale=="log":
        labelrotation=90.0
    else: 
        labelrotation=0.0
   
    axis.set_title(title, fontsize=fontsize)
    axis.set_xlabel(xlabel, fontsize=fontsize)
    axis.set_ylabel(ylabel, fontsize=fontsize)
    axis.tick_params(axis="x", labelsize=fontsize)
    axis.tick_params(axis="y", labelsize=fontsize, labelrotation=labelrotation)
    axis.set_yscale(yscale)
    axis.set_xscale(xscale)

    if xlim != None:
        axis.set_xlim(xlim[0], xlim[1])
    #axis.set_xticks(xticks)
    #axis.set_xticklabels(xticks_labels, fontsize=fontsize)


def compute_rank(results, axis=0, ascending=True):

    results = np.array(results)
    if ascending: 
        factor = -1

    else: 
        factor = 1
    
    n_items = results.shape[0]

    rank = stats.rankdata(factor*results, axis=axis)
    
    #cirtical=imputation of not valid entries
    #rank[np.isnan(results)] = np.sum(np.arange(n_items+1))/n_items
    rank[np.isnan(results)] = np.nan
    

    return rank

def draw_cd_diagram(ranks, experiments, bo_iter=100, name="Rank", path=None):

   
    df = pd.DataFrame(np.array(ranks )[:,:,bo_iter].T.tolist()).T
    df.columns = experiments
    df = df.stack().reset_index()
    df.columns = ["dataset_name", "classifier_name", "accuracy"]
    df.accuracy = -df.accuracy
    draw(df, path_name= path+name+".png", title=name)



optimizers = ["RS", "SMAC", "PBT"]
search_spaces = ["A2C", "PPO"]
environments = ["Pong-v0", "Alien-v0", "BankHeist-v0", "BeamRider-v0", "Breakout-v0", "Enduro-v0", "Phoenix-v0",
                                "Seaquest-v0", "SpaceInvaders-v0", "Riverraid-v0", "Tennis-v0", "Skiing-v0", "Boxing-v0", 
                                "Bowling-v0", "Asteroids-v0", "Ant-v2", "Hopper-v2", "Humanoid-v2", "CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v0"]
seeds = ["0","1","2"]
#seeds = ["0"]

data_path = ""
benchmark = BenchmarkHandler(data_path=data_path)
b_type ="wallclock"
linewidth = 3
fontsize = 20
plot_type = "DDPG"
plot_type = "3x2"
plot_type = "ARLBench"
plot_type = "per_env"
plot_type = "per_env_DDPG"

scale = "log"
xlim = (1000, None)
plt.tight_layout()

plt.rcParams.update({
    #"font.family": "serif",
    "font.size": fontsize
    #"text.usetex": True,
    #"pgf.texsystem": 'pdflatex', # default is xetex

})
    



if plot_type == "3x2":
    optimizers = ["RS", "SMAC", "PBT", "PB2"]
    for search_space in search_spaces:

        fig, axis = plt.subplots(1,3, figsize=(15,5))
        fig2, axis2 = plt.subplots(1,3, figsize=(15,5))

        for i, group in enumerate(benchmark.get_environments_groups()):

            
            environments = benchmark.get_environments_per_group (group) 
            results = []
            original_results = []
            budgets = []
            #search_space = search_spaces[0]
            budget_ranges = []
            to_omit_aggregated = []
            for optimizer in optimizers:

                collected_results, to_omit = collect_results(optimizer, search_space, environments, seeds)
                to_omit_aggregated.extend(to_omit)

                incumbents = np.array(collected_results["incumbents"])
                wallclock = np.array(collected_results["wallclock_time_eval"])
                timesteps = np.array(collected_results["timesteps_eval"])

                if b_type == "steps":
                    budget = timesteps
                elif b_type == "wallclock":
                    budget = wallclock
                else: 
                    budget = np.repeat( np.arange(100).reshape(1,-1), incumbents.shape[0], axis=0)

                budget_range = (np.nanmin(budget), np.nanmax(budget))
                budget_ranges.append(budget_range)


                #budget_range = (9000.0, 100000000.0)
                
                #budget_range = (16.596892595291138, 757445.3536088467) #A2C, wallclock
                #print(budget_range)

            budget_range = (np.array(budget_ranges).min(), np.array(budget_ranges).max())
            
            #ix = optimizers.index("SMAC")
            #budget_range = budget_ranges[ix]
            #xlim = budget_ranges[ix]
            


            for optimizer in optimizers:

                collected_results, _ = collect_results(optimizer, search_space, environments, seeds, to_omit_aggregated)


                incumbents = np.array(collected_results["incumbents"])
                wallclock = np.array(collected_results["wallclock_time_eval"])
                timesteps = np.array(collected_results["timesteps_eval"])

                if b_type == "steps":
                    budget = timesteps
                elif b_type == "wallclock":
                    budget = wallclock
                else: 
                    budget = np.repeat( np.arange(100).reshape(1,-1), incumbents.shape[0], axis=0)

                incumbents_on_grid, budget_on_grid =  discretized_grid (incumbents, budget, budget_range)

                results.append(incumbents_on_grid)
                budgets.append(budget_on_grid)
                last_reward = [incumbents[i][~np.isnan(incumbents[i])][-1:].item() for i in range(incumbents.shape[0])]
                original_results.append(last_reward)

            ranks=compute_rank(results)
            results = np.array(results)
            results_min = np.nanmin(results, axis=1)[:,np.newaxis,:]
            results_max = np.nanmax(results, axis=1)[:,np.newaxis,:]

            regret = (results_max-results)/(results_max-results_min)

            ranks  = ranks.transpose((1,0,2))
            regret  = regret.transpose((1,0,2))

            if i==0:
                title1 = group
                title2 = group
                ylabel1 = "Average Rank"
                ylabel2 = "Average Normalized Regret"
                xlabel1 = "Wallclock Time"
                xlabel2 = "Wallclock Time"

            elif i ==2:
                title1 = group
                title2 = group
                ylabel1 = ""
                ylabel2 = ""
                xlabel1 = "Wallclock Time"
                xlabel2 = "Wallclock Time"
            else:
                title1 = group
                title2 = group
                ylabel1 = ""
                ylabel2 = ""
                xlabel1 = "Wallclock Time"
                xlabel2 = "Wallclock Time"     
            plots_on_axis(axis[i], ranks, budget_on_grid.reshape(-1), draw_std=True, xscale=scale, legends=optimizers, fontsize=fontsize, linewidth=linewidth, xlabel=xlabel1, ylabel=ylabel1, title=title1,xlim=xlim)
            plots_on_axis(axis2[i], regret , budget_on_grid.reshape(-1), draw_std=True, xscale=scale, legends=optimizers, fontsize=fontsize, linewidth=linewidth, xlabel=xlabel2, ylabel=ylabel2, title=title2, xlim=xlim)


            print("CD plot")
            #subset_original_results = np.array([x[:,-1:] for x in original_results])
            original_rank = compute_rank(original_results)
            original_rank  = original_rank[:,:,np.newaxis].transpose((1,0,2))
            draw_cd_diagram(original_rank, optimizers, bo_iter=0, name=f"Rank {search_space} {group}", path="plots/")

        anchor = (0.25, 0, 0, -0.15)

        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:1]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc="center left", bbox_to_anchor=anchor, ncol=5, fontsize=fontsize)
        fig2.legend(lines, labels, loc="center left", bbox_to_anchor=anchor, ncol=5, fontsize=fontsize)

        plt.show()
        fig.savefig(f"plots/rank_results_{search_space}.pdf", bbox_inches="tight")
        fig2.savefig(f"plots/reward_results_{search_space}.pdf", bbox_inches="tight")


        fig.savefig(f"plots/baselines_results_{search_space}.png", bbox_inches="tight")


elif plot_type == "per_env_DDPG":

    continuous_env = ["Ant-v2", "Hopper-v2", "Humanoid-v2", "Pendulum-v0"]
    optimizers = ["RS", "SMAC", "PBT", "PB2"]
    search_spaces = ["DDPG"]

    for search_space in search_spaces:

        #fig, axis = plt.subplots(2,2, figsize=(10,10))
        #fig2, axis2 = plt.subplots(2,2, figsize=(10,10))
        fig, axis = plt.subplots(1,4, figsize=(20,5))
        fig2, axis2 = plt.subplots(1,4, figsize=(20,5))


        for i, environment in enumerate(continuous_env):

            
            #environments = benchmark.get_environments_per_group (group) 
            results = []
            original_results = []
            budgets = []
            #search_space = search_spaces[0]
            budget_ranges = []
            to_omit_aggregated = []
            for optimizer in optimizers:

                collected_results, to_omit = collect_results(optimizer, search_space, environments, seeds)
                to_omit_aggregated.extend(to_omit)

                incumbents = np.array(collected_results["incumbents"])
                wallclock = np.array(collected_results["wallclock_time_eval"])
                timesteps = np.array(collected_results["timesteps_eval"])

                if b_type == "steps":
                    budget = timesteps
                elif b_type == "wallclock":
                    budget = wallclock
                else: 
                    budget = np.repeat( np.arange(100).reshape(1,-1), incumbents.shape[0], axis=0)

                budget_range = (np.nanmin(budget), np.nanmax(budget))
                budget_ranges.append(budget_range)


                #budget_range = (9000.0, 100000000.0)
                
                #budget_range = (16.596892595291138, 757445.3536088467) #A2C, wallclock
                #print(budget_range)

            budget_range = (np.array(budget_ranges).min(), np.array(budget_ranges).max())
            
            #ix = optimizers.index("SMAC")
            #budget_range = budget_ranges[ix]
            #xlim = budget_ranges[ix]
            


            for optimizer in optimizers:

                if environment=="All (regret)" or environment=="All (rank)":
                    env_list = benchmark.get_environments()
                else:
                    env_list = [environment]

                collected_results, _ = collect_results(optimizer, search_space, env_list, seeds, to_omit_aggregated)


                incumbents = np.array(collected_results["incumbents"])
                wallclock = np.array(collected_results["wallclock_time_eval"])
                timesteps = np.array(collected_results["timesteps_eval"])

                if b_type == "steps":
                    budget = timesteps
                elif b_type == "wallclock":
                    budget = wallclock
                else: 
                    budget = np.repeat( np.arange(100).reshape(1,-1), incumbents.shape[0], axis=0)

                incumbents_on_grid, budget_on_grid =  discretized_grid (incumbents, budget, budget_range)

                results.append(incumbents_on_grid)
                budgets.append(budget_on_grid)
                last_reward = [incumbents[i][~np.isnan(incumbents[i])][-1:].item() for i in range(incumbents.shape[0])]
                original_results.append(last_reward)

            ranks=compute_rank(results)
            results = np.array(results)
            results_min = np.nanmin(results, axis=1)[:,np.newaxis,:]
            results_max = np.nanmax(results, axis=1)[:,np.newaxis,:]

            regret = (results_max-results)/(results_max-results_min)

            ranks  = ranks.transpose((1,0,2))
            regret  = regret.transpose((1,0,2))

            i1 = i//1
            i2 = i%1

            if i1 == 0:

                title1 = environment
                title2 = environment
                xlabel1 = "Wallclock Time"
                xlabel2 = "Wallclock Time"
        
            else:
  
                title1 = environment
                title2 = environment
                xlabel1 = ""
                xlabel2 = ""              

            yscale = "linear"
            if i2 == 0:
                ylabel1 = "Average Rank"
                ylabel2 = "Average Normalized Regret"

            if environment=="All (regret)":
                ranks = regret
                yscale = "log"
                
                
            if environment== "All (rank)":
                regret = ranks
            
            plots_on_axis(axis[i1], ranks, budget_on_grid.reshape(-1), draw_std=True, xscale=scale, legends=optimizers, fontsize=fontsize, linewidth=linewidth, xlabel=xlabel1, ylabel=ylabel1, title=title1,xlim=xlim, yscale=yscale)
            plots_on_axis(axis2[i1], regret , budget_on_grid.reshape(-1), draw_std=True, xscale=scale, legends=optimizers, fontsize=fontsize, linewidth=linewidth, xlabel=xlabel2, ylabel=ylabel2, title=title2, xlim=xlim, yscale="log")


            print("CD plot")
            #subset_original_results = np.array([x[:,-1:] for x in original_results])
            original_rank = compute_rank(original_results)
            original_rank  = original_rank[:,:,np.newaxis].transpose((1,0,2))
            #draw_cd_diagram(original_rank, optimizers, bo_iter=0, name=f"Rank {search_space} {group}", path="plots/")

        #anchor = (0.12, 0, 0, -0.02)
        anchor = (0.32, 0, 0, -0.15)
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:1]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc="center left", bbox_to_anchor=anchor, ncol=5, fontsize=fontsize)
        fig2.legend(lines, labels, loc="center left", bbox_to_anchor=anchor, ncol=5, fontsize=fontsize)
        plt.tight_layout()
        plt.show()
        fig.savefig(f"plots/rank_results_{search_space}_per_env.pdf", bbox_inches="tight")
        fig2.savefig(f"plots/reward_results_{search_space}_per_env.pdf", bbox_inches="tight")


        fig.savefig(f"plots/baselines_results_{search_space}_per_env.png", bbox_inches="tight")
elif plot_type == "per_env":

    optimizers = ["RS", "SMAC", "PBT", "PB2"]
    for search_space in search_spaces:

        fig, axis = plt.subplots(6,4, figsize=(20,30))
        fig2, axis2 = plt.subplots(6,4, figsize=(20,30))

        for i, environment in enumerate(benchmark.get_environments()+["All (rank)", "All (regret)"]):

            
            #environments = benchmark.get_environments_per_group (group) 
            results = []
            original_results = []
            budgets = []
            #search_space = search_spaces[0]
            budget_ranges = []
            to_omit_aggregated = []
            for optimizer in optimizers:

                collected_results, to_omit = collect_results(optimizer, search_space, environments, seeds)
                to_omit_aggregated.extend(to_omit)

                incumbents = np.array(collected_results["incumbents"])
                wallclock = np.array(collected_results["wallclock_time_eval"])
                timesteps = np.array(collected_results["timesteps_eval"])

                if b_type == "steps":
                    budget = timesteps
                elif b_type == "wallclock":
                    budget = wallclock
                else: 
                    budget = np.repeat( np.arange(100).reshape(1,-1), incumbents.shape[0], axis=0)

                budget_range = (np.nanmin(budget), np.nanmax(budget))
                budget_ranges.append(budget_range)


                #budget_range = (9000.0, 100000000.0)
                
                #budget_range = (16.596892595291138, 757445.3536088467) #A2C, wallclock
                #print(budget_range)

            budget_range = (np.array(budget_ranges).min(), np.array(budget_ranges).max())
            
            #ix = optimizers.index("SMAC")
            #budget_range = budget_ranges[ix]
            #xlim = budget_ranges[ix]
            


            for optimizer in optimizers:

                if environment=="All (regret)" or environment=="All (rank)":
                    env_list = benchmark.get_environments()
                else:
                    env_list = [environment]

                collected_results, _ = collect_results(optimizer, search_space, env_list, seeds, to_omit_aggregated)


                incumbents = np.array(collected_results["incumbents"])
                wallclock = np.array(collected_results["wallclock_time_eval"])
                timesteps = np.array(collected_results["timesteps_eval"])

                if b_type == "steps":
                    budget = timesteps
                elif b_type == "wallclock":
                    budget = wallclock
                else: 
                    budget = np.repeat( np.arange(100).reshape(1,-1), incumbents.shape[0], axis=0)

                incumbents_on_grid, budget_on_grid =  discretized_grid (incumbents, budget, budget_range)

                results.append(incumbents_on_grid)
                budgets.append(budget_on_grid)
                last_reward = [incumbents[i][~np.isnan(incumbents[i])][-1:].item() for i in range(incumbents.shape[0])]
                original_results.append(last_reward)

            ranks=compute_rank(results)
            results = np.array(results)
            results_min = np.nanmin(results, axis=1)[:,np.newaxis,:]
            results_max = np.nanmax(results, axis=1)[:,np.newaxis,:]

            regret = (results_max-results)/(results_max-results_min)

            ranks  = ranks.transpose((1,0,2))
            regret  = regret.transpose((1,0,2))

            i1 = i//4
            i2 = i%4

            if i1 == 5:

                title1 = environment
                title2 = environment
                xlabel1 = "Wallclock Time"
                xlabel2 = "Wallclock Time"
        
            else:
  
                title1 = environment
                title2 = environment
                xlabel1 = ""
                xlabel2 = ""              

            yscale = "linear"
            if i2 == 0:
                ylabel1 = "Average Rank"
                ylabel2 = "Average Normalized Regret"

            if environment=="All (regret)":
                ranks = regret
                yscale = "log"
                
                
            if environment== "All (rank)":
                regret = ranks
            
            plots_on_axis(axis[i1, i2], ranks, budget_on_grid.reshape(-1), draw_std=True, xscale=scale, legends=optimizers, fontsize=fontsize, linewidth=linewidth, xlabel=xlabel1, ylabel=ylabel1, title=title1,xlim=xlim, yscale=yscale)
            plots_on_axis(axis2[i1, i2], regret , budget_on_grid.reshape(-1), draw_std=True, xscale=scale, legends=optimizers, fontsize=fontsize, linewidth=linewidth, xlabel=xlabel2, ylabel=ylabel2, title=title2, xlim=xlim, yscale=yscale)


            print("CD plot")
            #subset_original_results = np.array([x[:,-1:] for x in original_results])
            original_rank = compute_rank(original_results)
            original_rank  = original_rank[:,:,np.newaxis].transpose((1,0,2))
            #draw_cd_diagram(original_rank, optimizers, bo_iter=0, name=f"Rank {search_space} {group}", path="plots/")

        anchor = (0.32, 0, 0, 0.12)

        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:1]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc="center left", bbox_to_anchor=anchor, ncol=5, fontsize=fontsize)
        fig2.legend(lines, labels, loc="center left", bbox_to_anchor=anchor, ncol=5, fontsize=fontsize)
        plt.tight_layout()
        plt.show()
        fig.savefig(f"plots/rank_results_{search_space}_per_env.pdf", bbox_inches="tight")
        fig2.savefig(f"plots/reward_results_{search_space}_per_env.pdf", bbox_inches="tight")


        fig.savefig(f"plots/baselines_results_{search_space}_per_env.png", bbox_inches="tight")


elif plot_type == "DDPG":

    search_space = "DDPG"
    optimizers = ["RS", "SMAC", "PBT", "PB2"]
    fig, axis = plt.subplots( figsize=(5,5))
    fig2, axis2 = plt.subplots( figsize=(5,5))
   
    environments = benchmark.get_environments()
    results = []
    original_results = []
    budgets = []
    #search_space = search_spaces[0]
    budget_ranges = []
    to_omit_aggregated = []

    for optimizer in optimizers:

        collected_results, to_omit = collect_results(optimizer, search_space, environments, seeds)
        to_omit_aggregated.extend(to_omit)

        incumbents = np.array(collected_results["incumbents"])
        wallclock = np.array(collected_results["wallclock_time_eval"])
        timesteps = np.array(collected_results["timesteps_eval"])

        if b_type == "steps":
            budget = timesteps
        elif b_type == "wallclock":
            budget = wallclock
        else: 
            budget =  np.arange(100)


        budget_range = (np.nanmin(budget), np.nanmax(budget))
        budget_ranges.append(budget_range)


        #budget_range = (9000.0, 100000000.0)
        
        #budget_range = (16.596892595291138, 757445.3536088467) #A2C, wallclock
        #print(budget_range)

    budget_range = (np.array(budget_ranges).min(), np.array(budget_ranges).max())



    for optimizer in optimizers:

        collected_results, _ = collect_results(optimizer, search_space, environments, seeds, to_omit_aggregated)
        incumbents = np.array(collected_results["incumbents"])
        wallclock = np.array(collected_results["wallclock_time_eval"])
        timesteps = np.array(collected_results["timesteps_eval"])
        budget = timesteps if b_type == "steps" else wallclock
        incumbents_on_grid, budget_on_grid =  discretized_grid (incumbents, budget, budget_range)

        results.append(incumbents_on_grid)
        budgets.append(budget_on_grid)
        original_results.append(incumbents)

    ranks=compute_rank(results)
    results = np.array(results)
    results_min = np.nanmin(results, axis=1)[:,np.newaxis,:]
    results_max = np.nanmax(results, axis=1)[:,np.newaxis,:]

    regret = (results_max-results)/(results_max-results_min)

    ranks  = ranks.transpose((1,0,2))
    regret  = regret.transpose((1,0,2))

    ylabel1 = "Average Rank"
    ylabel2 = "Average Normalized Regret"
    xlabel1 = "Wallclock Time"
    xlabel2 = "Wallclock Time"
    title1 = ""
    title2 = ""

    plots_on_axis(axis, ranks, budget_on_grid.reshape(-1), draw_std=True, xscale="log", legends=optimizers, fontsize=fontsize, linewidth=linewidth, xlabel=xlabel1, ylabel=ylabel1, title=title1, xlim=xlim)
    plots_on_axis(axis2, regret , budget_on_grid.reshape(-1), draw_std=True, xscale="log", legends=optimizers, fontsize=fontsize, linewidth=linewidth, xlabel=xlabel2, ylabel=ylabel2, title=title2, xlim=xlim)


    #anchor = (0.3, 0, 0, -0.4)
    anchor = (0.9, 0, 0.5, 1)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:1]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="center left", bbox_to_anchor=anchor, ncol=1, fontsize=fontsize)
    fig2.legend(lines, labels, loc="center left", bbox_to_anchor=anchor, ncol=1, fontsize=fontsize)

    plt.show()
    fig.savefig(f"plots/rank_results_{search_space}.pdf", bbox_inches="tight")
    fig2.savefig(f"plots/regret_results_{search_space}.pdf", bbox_inches="tight")

    fig.savefig(f"plots/baselines_results_{search_space}.png", bbox_inches="tight")



elif plot_type=="ARLBench":


    search_space = "PPO"
    optimizers = ["RS", "D-PBT", "D-PB2", "PBT", "PB2", "SMAC"]

    fig, axis = plt.subplots( figsize=(5,5))
    fig2, axis2 = plt.subplots( figsize=(5,5))

    environments = benchmark.get_environments()
    results = []
    original_results = []
    budgets = []
    #search_space = search_spaces[0]
    budget_ranges = []
    to_omit_aggregated = []

    for optimizer in optimizers:

        #if optimizer in ["PBT", "PB2"]:
        #    optimizer_path = optimizer+"_ARLBench"
        #else:
        optimizer_path = None

        collected_results, to_omit = collect_results(optimizer, search_space, environments, seeds, optimizer_path=optimizer_path)
        to_omit_aggregated.extend(to_omit)

        incumbents = np.array(collected_results["incumbents"])
        wallclock = np.array(collected_results["wallclock_time_eval"])
        timesteps = np.array(collected_results["timesteps_eval"])
        if b_type == "steps":
            budget = timesteps
        elif b_type == "wallclock":
            budget = wallclock
        else: 
            budget =  np.arange(100)


        budget_range = (np.nanmin(budget), np.nanmax(budget))
        budget_ranges.append(budget_range)


        #budget_range = (9000.0, 100000000.0)
        
        #budget_range = (16.596892595291138, 757445.3536088467) #A2C, wallclock
        #print(budget_range)

    budget_range = (np.array(budget_ranges).min(), np.array(budget_ranges).max())



    for optimizer in optimizers:

        #if optimizer in ["PBT", "PB2"]:
        #    optimizer_path = optimizer+"_ARLBench"
        #else:
        optimizer_path = None
        collected_results, _ = collect_results(optimizer, search_space, environments, seeds, to_omit_aggregated, optimizer_path=optimizer_path)
        incumbents = np.array(collected_results["incumbents"])
        wallclock = np.array(collected_results["wallclock_time_eval"])
        timesteps = np.array(collected_results["timesteps_eval"])

        if b_type == "steps":
            budget = timesteps
        elif b_type == "wallclock":
            budget = wallclock
        else: 
            budget =  np.arange(100)

        incumbents_on_grid, budget_on_grid =  discretized_grid (incumbents, budget, budget_range)

        results.append(incumbents_on_grid)
        budgets.append(budget_on_grid)
        original_results.append(incumbents)

    ranks=compute_rank(results)
    results = np.array(results)
    results_min = np.nanmin(results, axis=1)[:,np.newaxis,:]
    results_max = np.nanmax(results, axis=1)[:,np.newaxis,:]

    regret = (results_max-results)/(results_max-results_min)

    ranks  = ranks.transpose((1,0,2))
    regret  = regret.transpose((1,0,2))

    ylabel1 = "Average Rank"
    ylabel2 = "Average Normalized Regret"
    xlabel1 = "Wallclock Time"
    xlabel2 = "Wallclock Time"
    title1 = ""
    title2 = ""

    plots_on_axis(axis, ranks, budget_on_grid.reshape(-1), draw_std=True, xscale="log", legends=optimizers, fontsize=fontsize, linewidth=linewidth, xlabel=xlabel1, ylabel=ylabel1, title=title1, xlim = xlim)
    plots_on_axis(axis2, regret , budget_on_grid.reshape(-1), draw_std=True, xscale="log", legends=optimizers, fontsize=fontsize, linewidth=linewidth, xlabel=xlabel2, ylabel=ylabel2, title=title2, xlim= xlim)

    anchor = (0.25, 0, 0, -0.65)
    anchor = (0.9, 0, 0.5, 1)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:1]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="center left", bbox_to_anchor=anchor, ncol=1, fontsize=fontsize)
    fig2.legend(lines, labels, loc="center left", bbox_to_anchor=anchor, ncol=1, fontsize=fontsize)

    plt.show()
    fig.savefig(f"plots/rank_results_{search_space}_ARL.pdf", bbox_inches="tight")
    fig2.savefig(f"plots/regret_results_{search_space}_ARL.pdf", bbox_inches="tight") 
    
    fig.savefig(f"plots/baselines_results_{search_space}_ARL.png", bbox_inches="tight")