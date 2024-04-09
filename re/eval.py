from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import bernoulli
from tqdm.notebook import tqdm, trange
from pyecsca.misc.utils import TaskExecutor


errs = (0, 0.1, 0.2, 0.3, 0.4, 0.5)
majs = (1, 3, 5, 7, 9, 11)

nums = (4, 10, 20, 40, 60) 
smpls = (1, 2, 3, 5, 10)

def walk_symmetric(tree, err, majority, cfg):
    current = tree.root
    B = bernoulli(err)
    queries = 0
    while not current.is_leaf:
        dmap_index = current.dmap_index
        dmap_input = current.dmap_input
        dmap = tree.maps[dmap_index]
        true_response = dmap[cfg, dmap_input]
        responses = []
        response = None
        for _ in range(majority):
            responses.append(true_response ^ B.rvs())
            if responses.count(True) > (majority // 2):
                response = True
                break
            if responses.count(False) > (majority // 2):
                response = False
                break
        response_map = {child.response: child for child in current.children}
        current = response_map[response]
        queries += len(responses)
    return cfg in current.cfgs, len(current.cfgs), queries


def _eval_symmetric(tree, cfg, errs, majs, num_tries):
    correct_tries = np.zeros((len(errs), len(majs)))
    precise_tries = np.zeros((len(errs), len(majs)))
    amount_tries = np.zeros((len(errs), len(majs)))
    query_tries = np.zeros((len(errs), len(majs)))
    for i, err in enumerate(errs):
        for j, majority in enumerate(majs):
            for _ in range(num_tries):
                correct, amount, queries = walk_symmetric(tree, err, majority, cfg)
                correct_tries[i, j] += correct
                precise_tries[i, j] += (amount == 1)
                amount_tries[i, j] += amount
                query_tries[i, j] += queries
    return correct_tries, precise_tries, amount_tries, query_tries

    
def eval_tree_symmetric(cfgs, build_tree, num_trees, num_tries, num_cores):
    correct_tries = np.zeros((len(errs), len(majs)))
    precise_tries = np.zeros((len(errs), len(majs)))
    amount_tries = np.zeros((len(errs), len(majs)))
    query_tries = np.zeros((len(errs), len(majs)))

    trees = []
    with TaskExecutor(max_workers=num_cores) as pool:
        for i in range(num_trees):
            # Build the trees
            pool.submit_task((i,), build_tree, cfgs)
        for (i,), future in tqdm(pool.as_completed(), total=len(pool.tasks), desc="Building trees", smoothing=0):
            trees.append(future.result())

    with TaskExecutor(max_workers=num_cores) as pool:
        for i, tree in enumerate(trees):
            for cfg in cfgs:
                # Now cfg is the "true" config
                pool.submit_task((i, cfg), _eval_symmetric, tree, cfg, errs, majs, num_tries)
        for (i, cfg), future in tqdm(pool.as_completed(), total=len(pool.tasks), desc="Computing", smoothing=0):
            c_tries, p_tries, a_tries, q_tries = future.result()
            correct_tries += c_tries
            precise_tries += p_tries
            amount_tries += a_tries
            query_tries += q_tries
    total = num_trees * num_tries * len(cfgs)

    correct_rate = (correct_tries * 100) / total
    precise_rate = (precise_tries * 100) / total
    amount_rate = amount_tries / total
    query_rate = query_tries / total
    return correct_rate[...,::-1], precise_rate[...,::-1], amount_rate[...,::-1], query_rate[...,::-1]


def walk_asymmetric(tree, err_0, err_1, majority, cfg):
    current = tree.root
    B0 = bernoulli(err_0)
    B1 = bernoulli(err_1)
    queries = 0
    while not current.is_leaf:
        dmap_index = current.dmap_index
        dmap_input = current.dmap_input
        dmap = tree.maps[dmap_index]
        true_response = dmap[cfg, dmap_input]
        responses = []
        response = None
        for _ in range(majority):
            responses.append(true_response ^ (B1.rvs() if true_response else B0.rvs()))
            if responses.count(True) > (majority // 2):
                response = True
                break
            if responses.count(False) > (majority // 2):
                response = False
                break
        response_map = {child.response: child for child in current.children}
        current = response_map[response]
        queries += len(responses)
    return cfg in current.cfgs, len(current.cfgs), queries


def _eval_asymmetric(tree, cfg, errs, majs, num_tries):
    correct_tries = np.zeros((len(errs), len(errs), len(majs)))
    precise_tries = np.zeros((len(errs), len(errs), len(majs)))
    amount_tries =  np.zeros((len(errs), len(errs), len(majs)))
    query_tries =   np.zeros((len(errs), len(errs), len(majs)))

    for i, err_0 in enumerate(errs):
        for j, err_1 in enumerate(errs):
            for k, majority in enumerate(majs):
                for _ in range(num_tries):
                    correct, amount, queries = walk_asymmetric(tree, err_0, err_1, majority, cfg)
                    correct_tries[i, j, k] += correct
                    precise_tries[i, j, k] += (amount == 1)
                    amount_tries[i, j, k] += amount
                    query_tries[i, j, k] += queries
    return correct_tries, precise_tries, amount_tries, query_tries


def eval_tree_asymmetric(cfgs, build_tree, num_trees, num_tries, num_cores):
    correct_tries = np.zeros((len(errs), len(errs), len(majs)))
    precise_tries = np.zeros((len(errs), len(errs), len(majs)))
    amount_tries =  np.zeros((len(errs), len(errs), len(majs)))
    query_tries =   np.zeros((len(errs), len(errs), len(majs)))

    trees = []
    with TaskExecutor(max_workers=num_cores) as pool:
        for i in range(num_trees):
            # Build the trees
            pool.submit_task((i,), build_tree, cfgs)
        for (i,), future in tqdm(pool.as_completed(), total=len(pool.tasks), desc="Building trees", smoothing=0):
            trees.append(future.result())

    with TaskExecutor(max_workers=num_cores) as pool:
        for i, tree in enumerate(trees):
            for cfg in cfgs:
                # Now cfg is the "true" config
                pool.submit_task((i, cfg), _eval_asymmetric, tree, cfg, errs, majs, num_tries)
        for (i, cfg), future in tqdm(pool.as_completed(), total=len(pool.tasks), desc="Computing", smoothing=0):
            c_tries, p_tries, a_tries, q_tries = future.result()
            correct_tries += c_tries
            precise_tries += p_tries
            amount_tries += a_tries
            query_tries += q_tries
    total = num_trees * num_tries * len(cfgs)

    correct_rate = (correct_tries * 100) / total
    precise_rate = (precise_tries * 100) / total
    amount_rate = amount_tries / total
    query_rate = query_tries / total
    return correct_rate, precise_rate, amount_rate, query_rate


def walk_binomial(tree, num, smpl, majority, cfg):
    current = tree.root
    B = binom(num, 0.5)
    queries = 0
    while not current.is_leaf:
        dmap_index = current.dmap_index
        dmap_input = current.dmap_input
        dmap = tree.maps[dmap_index]
        true_response = dmap[cfg, dmap_input]
        responses = [true_response + B.rvs() - (num // 2) for _ in range(smpl)]
        mean = np.mean(responses)
        response_map = {child.response: child for child in current.children}
        closest = min(response_map, key=lambda value: abs(value-mean))
        current = response_map[closest]
        queries += smpl
    return cfg in current.cfgs, len(current.cfgs), queries


def _eval_binomial(tree, cfg, nums, smpls, num_tries):
    correct_tries = np.zeros((len(nums), len(smpls)))
    precise_tries = np.zeros((len(nums), len(smpls)))
    amount_tries =  np.zeros((len(nums), len(smpls)))
    query_tries =   np.zeros((len(nums), len(smpls)))

    for i, num in enumerate(nums):
        for j, smpl in enumerate(smpls):
            for _ in range(num_tries):
                correct, amount, queries = walk_binomial(tree, num, smpl, cfg)
                correct_tries[i, j] += correct
                precise_tries[i, j] += (amount == 1)
                amount_tries[i, j] += amount
                query_tries[i, j] += queries
    return correct_tries, precise_tries, amount_tries, query_tries


def eval_tree_binomial(cfgs, build_tree, num_trees, num_tries, num_cores)
    correct_tries = np.zeros((len(nums), len(smpls)))
    precise_tries = np.zeros((len(nums), len(smpls)))
    amount_tries =  np.zeros((len(nums), len(smpls)))
    query_tries =   np.zeros((len(nums), len(smpls)))

    trees = []
    with TaskExecutor(max_workers=num_cores) as pool:
        for i in range(num_trees):
            # Build the trees
            pool.submit_task((i,), build_tree, cfgs)
        for (i,), future in tqdm(pool.as_completed(), total=len(pool.tasks), desc="Building trees", smoothing=0):
            trees.append(future.result())

    with TaskExecutor(max_workers=num_cores) as pool:
        for i, tree in enumerate(trees):
            for cfg in cfgs:
                # Now cfg is the "true" config
                pool.submit_task((i, cfg), _eval_binomial, tree, cfg, nums, smpls, num_tries)
        for (i, cfg), future in tqdm(pool.as_completed(), total=len(pool.tasks), desc="Computing", smoothing=0):
            c_tries, p_tries, a_tries, q_tries = future.result()
            correct_tries += c_tries
            precise_tries += p_tries
            amount_tries += a_tries
            query_tries += q_tries
    total = num_trees * num_tries * len(cfgs)

    correct_rate = (correct_tries * 100) / total
    precise_rate = (precise_tries * 100) / total
    amount_rate = amount_tries / total
    query_rate = query_tries / total
    return correct_rate, precise_rate, amount_rate, query_rate
            

def query_rate_symmetric(query_rate):
    fig, ax = plt.subplots()
    im = ax.imshow(query_rate.T, cmap="plasma")
    cbar_ax = fig.add_axes((0.85, 0.15, 0.04, 0.69))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Oracle query rate", rotation=-90, va="bottom")
    
    ax.set_xticks(np.arange(len(errs)), labels=errs)
    ax.set_yticks(np.arange(len(majs)), labels=reversed(majs))
    ax.set_xlabel("error probability")
    ax.set_ylabel("majority vote")
    for i in range(len(errs)):
        for j in range(len(majs)):
            text = ax.text(i, j, f"{query_rate[i, j]:.1f}",
                           ha="center", va="center", color="white" if i - j <= 2 else "black")
    return fig


def success_rate_symmetric(correct_rate, baseline):
    fig, ax = plt.subplots()
    im = ax.imshow(correct_rate.T, vmin=0, cmap="viridis")
    cbar_ax = fig.add_axes((0.85, 0.15, 0.04, 0.69))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Success rate", rotation=-90, va="bottom")
    cbar.ax.axhline(baseline, color="red", linestyle="--")
    
    ax.set_xticks(np.arange(len(errs)), labels=errs)
    ax.set_yticks(np.arange(len(majs)), labels=reversed(majs))
    ax.set_xlabel("error probability")
    ax.set_ylabel("majority vote")
    for i in range(len(errs)):
        for j in range(len(majs)):
            c_rate = correct_rate[i, j]
            text = ax.text(i, j, f"{c_rate:.1f}%",
                           ha="center", va="center", color="white" if c_rate < 80 else "black")
    return fig


def precise_rate_symmetric(precise_rate):
    fig, ax = plt.subplots()
    im = ax.imshow(precise_rate.T, vmin=0, cmap="viridis")
    cbar_ax = fig.add_axes((0.85, 0.15, 0.04, 0.69))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Precision", rotation=-90, va="bottom")
    
    ax.set_xticks(np.arange(len(errs)), labels=errs)
    ax.set_yticks(np.arange(len(majs)), labels=reversed(majs))
    ax.set_xlabel("error probability")
    ax.set_ylabel("majority vote")
    for i in range(len(errs)):
        for j in range(len(majs)):
            p_rate = precise_rate[i, j]
            text = ax.text(i, j, f"{p_rate:.1f}%",
                           ha="center", va="center", color="white" if p_rate < 80 else "black")
    return fig


def success_rate_vs_query_rate_symmetric(query_rate, correct_rate):
    fig, ax = plt.subplots()
    ax.grid()
    for i, err in enumerate(errs):
        qrs = query_rate[i, :]
        crs = correct_rate[i, :]
        ax.scatter(qrs, crs, label=f"error = {err}")
    ax.set_xlabel("oracle queries")
    ax.set_ylabel("success rate")
    ax.legend()
    return fig


def success_rate_vs_majority_symmetric(correct_rate):
    fig, ax = plt.subplots()
    ax.grid()
    for i, err in enumerate(errs):
        crs = correct_rate[i, :]
        ax.plot(list(reversed(majs)), crs, label=f"error = {err}")
    ax.set_xlabel("majority vote")
    ax.set_ylabel("success rate")
    ax.set_xticks(majs)
    ax.legend()
    return fig


def query_rate_asymmetric(query_rate_b):
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex="col", sharey="row")
    vmin = np.min(query_rate_b)
    vmax = np.max(query_rate_b)
    
    for row in range(2):
        for col in range(3):
            ax = axs[row, col]
            level = row * 3 + col
            im = ax.imshow(query_rate_b[::-1,:,level], cmap="plasma", vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(errs)), labels=errs)
            ax.set_yticks(np.arange(len(errs)), labels=list(reversed(errs)))
            for i in range(len(errs)):
                for j in range(len(errs)):
                    q = query_rate_b[i, len(errs) - j - 1, level]
                    q_rate = f"{q:.0f}"
                    text = ax.text(i, j, q_rate, ha="center", va="center", color="white" if q < (vmax - vmin)//2 else "black")
            ax.set_xlabel("$e_1$")
            ax.set_ylabel("$e_O$")
            ax.set_title(majs[level])
    fig.set_size_inches((10,6))
    fig.tight_layout(h_pad=1.5, rect=(0, 0, 0.9, 1))
    cbar_ax = fig.add_axes((0.9, 0.10, 0.02, 0.84))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Oracle query rate", rotation=-90, va="bottom")
    return fig


def success_rate_asymmetric(correct_rate_b, baseline):
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex="col", sharey="row")
    for row in range(2):
        for col in range(3):
            ax = axs[row, col]
            level = row * 3 + col
            im = ax.imshow(correct_rate_b[::-1,:,level], cmap="viridis", vmin=0, vmax=100)
            ax.set_xticks(np.arange(len(errs)), labels=errs)
            ax.set_yticks(np.arange(len(errs)), labels=list(reversed(errs)))
            for i in range(len(errs)):
                for j in range(len(errs)):
                    c = correct_rate_b[i, len(errs) - j - 1, level]
                    c_rate = f"{c:.0f}%"
                    text = ax.text(i, j, c_rate, ha="center", va="center", color="white" if c < 50 else "black")
            ax.set_xlabel("$e_1$")
            ax.set_ylabel("$e_O$")
            ax.set_title(majs[level])
    fig.set_size_inches((10,6))
    fig.tight_layout(h_pad=1.5, rect=(0, 0, 0.9, 1))
    cbar_ax = fig.add_axes((0.9, 0.10, 0.02, 0.84))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Success rate", rotation=-90, va="bottom")
    cbar.ax.axhline(baseline, color="red", linestyle="--")
    return fig


def precise_rate_asymmetric(precise_rate_b):
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex="col", sharey="row")
    for row in range(2):
        for col in range(3):
            ax = axs[row, col]
            level = row * 3 + col
            im = ax.imshow(precise_rate_b[::-1,:,level], cmap="viridis", vmin=0, vmax=100)
            ax.set_xticks(np.arange(len(errs)), labels=errs)
            ax.set_yticks(np.arange(len(errs)), labels=list(reversed(errs)))
            for i in range(len(errs)):
                for j in range(len(errs)):
                    p = precise_rate_b[i, len(errs) - j - 1, level]
                    p_rate = f"{p:.0f}%"
                    text = ax.text(i, j, p_rate, ha="center", va="center", color="white" if p < 80 else "black")
            ax.set_xlabel("$e_1$")
            ax.set_ylabel("$e_O$")
            ax.set_title(majs[level])
    fig.set_size_inches((10,6))
    fig.tight_layout(h_pad=1.5, rect=(0, 0, 0.9, 1))
    cbar_ax = fig.add_axes((0.9, 0.10, 0.02, 0.84))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Precision rate", rotation=-90, va="bottom")
    return fig


def success_rate_vs_majority_asymmetric(correct_rate_b):
    fig, ax = plt.subplots()
    ax.grid()
    crs_accumulated = {}
    for i, err_0 in enumerate(errs):
        for j, err_1 in enumerate(errs):
            crs = correct_rate_b[i, j, :]
            total_err = round(err_0 + err_1, 1)
            l = crs_accumulated.setdefault(total_err, [])
            l.append(crs)
    for total_err in crs_accumulated.keys():
        crs = np.mean(crs_accumulated[total_err], axis=0)
        ax.plot(majs, crs, label=f"total_error = {total_err}")
    ax.set_xticks(majs)
    ax.set_xlabel("majority")
    ax.set_ylabel("success rate")
    ax.legend(bbox_to_anchor=(1, 1.02))
    fig.tight_layout()
    return fig


def query_rate_binomial(query_rate):
    fig, ax = plt.subplots()
    im = ax.imshow(query_rate.T, cmap="plasma")
    cbar_ax = fig.add_axes((0.85, 0.15, 0.04, 0.69))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Oracle query rate", rotation=-90, va="bottom")
    
    ax.set_xticks(np.arange(len(nums)), labels=nums)
    ax.set_yticks(np.arange(len(smpls)), labels=reversed(smpls))
    ax.set_xlabel("binom n")
    ax.set_ylabel("samples")
    for i in range(len(nums)):
        for j in range(len(smpls)):
            text = ax.text(i, j, f"{query_rate[i, j]:.1f}",
                           ha="center", va="center", color="white" if i - j <= 2 else "black")
    return fig


def success_rate_binomial(correct_rate, baseline):
    fig, ax = plt.subplots()
    im = ax.imshow(correct_rate.T, vmin=0, cmap="viridis")
    cbar_ax = fig.add_axes((0.85, 0.15, 0.04, 0.69))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Success rate", rotation=-90, va="bottom")
    cbar.ax.axhline(baseline, color="red", linestyle="--")
    
    ax.set_xticks(np.arange(len(nums)), labels=nums)
    ax.set_yticks(np.arange(len(smpls)), labels=reversed(smpls))
    ax.set_xlabel("binom n")
    ax.set_ylabel("samples")
    for i in range(len(nums)):
        for j in range(len(smpls)):
            c_rate = correct_rate[i, j]
            text = ax.text(i, j, f"{c_rate:.1f}%",
                           ha="center", va="center", color="white" if c_rate < 80 else "black")
    return fig


def precise_rate_binomial(precise_rate):
    fig, ax = plt.subplots()
    im = ax.imshow(precise_rate.T, vmin=0, cmap="viridis")
    cbar_ax = fig.add_axes((0.85, 0.15, 0.04, 0.69))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Precision", rotation=-90, va="bottom")
    
    ax.set_xticks(np.arange(len(nums)), labels=nums)
    ax.set_yticks(np.arange(len(smpls)), labels=reversed(smpls))
    ax.set_xlabel("binom n")
    ax.set_ylabel("samples")
    for i in range(len(nums)):
        for j in range(len(smpls)):
            p_rate = precise_rate[i, j]
            text = ax.text(i, j, f"{p_rate:.1f}%",
                           ha="center", va="center", color="white" if p_rate < 80 else "black")
    return fig
