#%%

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from einops import rearrange
from config import DatasetConfig, DataKey, MetaKey
from analyze_utils import prep_plt

#%%

# Temperature stuff
palette = sns.color_palette('colorblind', 2)
from torch.distributions import poisson

lambdas = torch.arange(0, 30, 2)
def change_temp(probs, temperature):
    return (probs / temperature).exp()  / (probs / temperature).exp().sum()
for l in lambdas:
    dist = poisson.Poisson(l)
    plt.plot(dist.log_prob(torch.arange(0, 20)).exp(), color=palette[0])
    plt.plot(change_temp(dist.log_prob(torch.arange(0, 20)).exp(), 0.01), color=palette[1])

#%%
batch = torch.load('valid.pth')
# batch = torch.load('debug_batch.pth')

#%%
print(batch['tgt'].size())
sns.histplot(batch['tgt'].cpu().numpy().flatten())
#%%
print(batch[DataKey.spikes].size())
print(batch[DataKey.bhvr_vel].size())

trial = 0
# trial = 1
# trial = 2
# trial = 3
# trial = 4
# trial = 5

trial_vel = batch[DataKey.bhvr_vel][trial].cpu()
trial_spikes = batch[DataKey.spikes][trial].cpu()

def plot_vel(vel, ax):
    ax = prep_plt(ax=ax)
    ax.plot(vel)
def plot_raster(spikes, ax, vert_space=0.1, bin_size_ms=5):
    ax = prep_plt(ax)
    spikes = rearrange(spikes, 't a c h -> t (a c h)')
    sns.despine(ax=ax, left=True, bottom=False)
    spike_t, spike_c = np.where(spikes)
    # prep_plt(axes[_c], big=True)
    time = np.arange(spikes.shape[0])
    ax.scatter(
        time[spike_t], spike_c * vert_space,
        # c=colors,
        marker='|',
        s=10,
        alpha=0.9
        # alpha=0.3
    )
    time_lim = spikes.shape[0] * bin_size_ms
    ax.set_xticks(np.linspace(0, spikes.shape[0], 5))
    ax.set_xticklabels(np.linspace(0, time_lim, 5))
    # ax.set_title("Benchmark Maze (Sorted)")
    # ax.set_title(context.alias)
    ax.set_xlabel('Time (ms)')
    ax.set_yticks([])
    return ax


f = plt.figure(figsize=(10, 10))
plot_vel(trial_vel, f.add_subplot(2, 1, 1))
plot_raster(trial_spikes, f.add_subplot(2, 1, 2), bin_size_ms=20)



#%%
# Draw a 3d scatterplot of several random point clouds in space
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

colors = sns.color_palette('colorblind', 10)
fig = plt.figure()
ax = fig.add_subplot(111)
# ax = fig.add_subplot(111, projection='3d')

# Generate random data
n = 100
xs = np.random.rand(n)
ys = np.random.rand(n)
zs = np.random.rand(n)

def plot_3d_cluster(mu, std, color):
    xs, ys, zs = generate_3d_cluster(mu, std)
    ax.scatter(xs, ys, zs, c=color, marker='o')

def generate_2d_cluster(mu, std):
    # Generate random vector for array inputs mu, std
    n = 100
    xs = np.random.normal(mu[0], std[0], n)
    ys = np.random.normal(mu[1], std[1], n)
    return xs, ys

def plot_2d_cluster(mu, std, color):
    xs, ys = generate_2d_cluster(mu, std)
    ax.scatter(xs, ys, c=color, marker='o')
# Plot the points
# ax.scatter(xs, ys, zs, c='b', marker='o')
mus = np.random.rand(2, 10) * 10
stds = np.random.rand(2, 10)
for i in range(5):
    plot_2d_cluster(mus[:, i], stds[:, i], color=colors[i])
