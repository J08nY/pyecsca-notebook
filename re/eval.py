from matplotlib import pyplot as plt
from matplotlib import colormaps
import numpy as np
import xarray as xr
import seaborn as sns
from scipy.stats import bernoulli, binom
from tqdm.notebook import tqdm, trange
from pyecsca.misc.utils import TaskExecutor


errs = (0, 0.1, 0.2, 0.3, 0.4, 0.5)
majs = (1, 3, 5, 7, 9, 11)

nums = (0, 4, 10, 20, 40, 60)
smpls = (1, 2, 3, 5, 10, 20)

plasma = colormaps["plasma"]
viridis = colormaps["viridis"]
mako = sns.color_palette("mako", as_cmap=True)


def _bins_array(name):
    return xr.DataArray(np.zeros((len(errs), len(majs))), dims=("err", "majority"), coords={"err": list(errs), "majority": list(majs)}, name=name)


def _bina_array(name):
    return xr.DataArray(np.zeros((len(errs), len(errs), len(majs))), dims=("err_0", "err_1", "majority"), coords={"err_0": list(errs), "err_1": list(errs), "majority": list(majs)}, name=name)


def _binom_array(name):
    return xr.DataArray(np.zeros((len(nums), len(smpls))), dims=("num", "sample"), coords={"num": list(nums), "sample": list(smpls)}, name=name)


def walk_symmetric(tree, err, majority, cfg):
    current = tree.root
    B = bernoulli(err)
    queries = 0
    while not current.is_leaf:
        response_map = {child.response: child for child in current.children}
        
        dmap_index = current.dmap_index
        dmap_input = current.dmap_input
        dmap = tree.maps[dmap_index]
        true_response = dmap[cfg, dmap_input]
        
        if set(dmap.codomain) not in ({True, False}, {True, False, None}):
            current = response_map[true_response]
            continue
        
        responses = []
        response = true_response
        for _ in range(majority):
            responses.append(true_response ^ B.rvs())
            if responses.count(True) > (majority // 2):
                response = True
                break
            if responses.count(False) > (majority // 2):
                response = False
                break
        current = response_map[response]
        queries += len(responses)
    return cfg in current.cfgs, len(current.cfgs), queries


def _eval_symmetric(tree, cfg, errs, majs, num_tries):
    correct_tries = _bins_array("correct")
    precise_tries = _bins_array("precise")
    amount_tries  = _bins_array("amount")
    query_tries   = _bins_array("query")

    for err in errs:
        for majority in majs:
            for _ in range(num_tries):
                pos = {"err": err, "majority": majority}
                correct, amount, queries = walk_symmetric(tree, err, majority, cfg)
                correct_tries.loc[pos] += correct
                precise_tries.loc[pos] += (amount == 1)
                amount_tries.loc[pos]  += amount
                query_tries.loc[pos]   += queries
    return correct_tries, precise_tries, amount_tries, query_tries


def eval_tree_symmetric1(cfgs, trees, num_tries, num_cores):
    correct_tries = _bins_array("correct")
    precise_tries = _bins_array("precise")
    amount_tries  = _bins_array("amount")
    query_tries   = _bins_array("query")

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
    total = len(trees) * num_tries * len(cfgs)

    correct_rate = (correct_tries * 100) / total
    precise_rate = (precise_tries * 100) / total
    amount_rate = amount_tries / total
    query_rate = query_tries / total
    return correct_rate, precise_rate, amount_rate, query_rate


def eval_tree_symmetric(cfgs, build_tree, num_trees, num_tries, num_cores):
    trees = []
    with TaskExecutor(max_workers=num_cores) as pool:
        for i in range(num_trees):
            # Build the trees
            pool.submit_task((i,), build_tree, cfgs)
        for (i,), future in tqdm(pool.as_completed(), total=len(pool.tasks), desc="Building trees", smoothing=0):
            trees.append(future.result())
    return eval_tree_symmetric1(cfgs, trees, num_tries, num_cores)


def walk_asymmetric(tree, err_0, err_1, majority, cfg):
    current = tree.root
    B0 = bernoulli(err_0)
    B1 = bernoulli(err_1)
    queries = 0
    while not current.is_leaf:
        response_map = {child.response: child for child in current.children}
        
        dmap_index = current.dmap_index
        dmap_input = current.dmap_input
        dmap = tree.maps[dmap_index]
        true_response = dmap[cfg, dmap_input]
        
        if set(dmap.codomain) not in ({True, False}, {True, False, None}):
            current = response_map[true_response]
            continue

        responses = []
        response = true_response
        for _ in range(majority):
            responses.append(true_response ^ (B1.rvs() if true_response else B0.rvs()))
            if responses.count(True) > (majority // 2):
                response = True
                break
            if responses.count(False) > (majority // 2):
                response = False
                break
        current = response_map[response]
        queries += len(responses)
    return cfg in current.cfgs, len(current.cfgs), queries


def _eval_asymmetric(tree, cfg, errs, majs, num_tries):
    correct_tries = _bina_array("correct")
    precise_tries = _bina_array("precise")
    amount_tries =  _bina_array("amount")
    query_tries =   _bina_array("query")

    for err_0 in errs:
        for err_1 in errs:
            for majority in majs:
                for _ in range(num_tries):
                    pos = {"err_0": err_0, "err_1": err_1, "majority": majority}
                    correct, amount, queries = walk_asymmetric(tree, err_0, err_1, majority, cfg)
                    correct_tries.loc[pos] += correct
                    precise_tries.loc[pos] += (amount == 1)
                    amount_tries.loc[pos]  += amount
                    query_tries.loc[pos]   += queries
    return correct_tries, precise_tries, amount_tries, query_tries


def eval_tree_asymmetric1(cfgs, trees, num_tries, num_cores):
    correct_tries = _bina_array("correct")
    precise_tries = _bina_array("precise")
    amount_tries =  _bina_array("amount")
    query_tries =   _bina_array("query")

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
    total = len(trees) * num_tries * len(cfgs)

    correct_rate = (correct_tries * 100) / total
    precise_rate = (precise_tries * 100) / total
    amount_rate = amount_tries / total
    query_rate = query_tries / total
    return correct_rate, precise_rate, amount_rate, query_rate


def eval_tree_asymmetric(cfgs, build_tree, num_trees, num_tries, num_cores):
    trees = []
    with TaskExecutor(max_workers=num_cores) as pool:
        for i in range(num_trees):
            # Build the trees
            pool.submit_task((i,), build_tree, cfgs)
        for (i,), future in tqdm(pool.as_completed(), total=len(pool.tasks), desc="Building trees", smoothing=0):
            trees.append(future.result())
    return eval_tree_asymmetric1(cfgs, trees, num_tries, num_cores)


def walk_binomial(tree, num, smpl, cfg):
    current = tree.root
    B = binom(num, 0.5)
    queries = 0
    while not current.is_leaf:
        response_map = {child.response: child for child in current.children}
        
        dmap_index = current.dmap_index
        dmap_input = current.dmap_input
        dmap = tree.maps[dmap_index]
        true_response = dmap[cfg, dmap_input]
        if dmap_input == "category": # ZVP/EPA special-case
            current = response_map[true_response]
            continue
            
        responses = [true_response + B.rvs() - (num // 2) for _ in range(smpl)]
        mean = np.mean(responses)
        closest = min(response_map, key=lambda value: abs(value-mean))
        current = response_map[closest]
        queries += smpl
    return cfg in current.cfgs, len(current.cfgs), queries


def _eval_binomial(tree, cfg, nums, smpls, num_tries):
    correct_tries = _binom_array("correct")
    precise_tries = _binom_array("correct")
    amount_tries =  _binom_array("correct")
    query_tries =   _binom_array("correct")

    for num in nums:
        for smpl in smpls:
            for _ in range(num_tries):
                pos = {"num": num, "sample": smpl}
                correct, amount, queries = walk_binomial(tree, num, smpl, cfg)
                correct_tries.loc[pos] += correct
                precise_tries.loc[pos] += (amount == 1)
                amount_tries.loc[pos] += amount
                query_tries.loc[pos] += queries
    return correct_tries, precise_tries, amount_tries, query_tries


def eval_tree_binomial1(cfgs, trees, num_tries, num_cores):
    correct_tries = _binom_array("correct")
    precise_tries = _binom_array("precise")
    amount_tries =  _binom_array("amount")
    query_tries =   _binom_array("query")

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
    total = len(trees) * num_tries * len(cfgs)

    correct_rate = (correct_tries * 100) / total
    precise_rate = (precise_tries * 100) / total
    amount_rate = amount_tries / total
    query_rate = query_tries / total
    return correct_rate, precise_rate, amount_rate, query_rate


def eval_tree_binomial(cfgs, build_tree, num_trees, num_tries, num_cores):
    trees = []
    with TaskExecutor(max_workers=num_cores) as pool:
        for i in range(num_trees):
            # Build the trees
            pool.submit_task((i,), build_tree, cfgs)
        for (i,), future in tqdm(pool.as_completed(), total=len(pool.tasks), desc="Building trees", smoothing=0):
            trees.append(future.result())
    return eval_tree_binomial1(cfgs, trees, num_tries, num_cores)


def _text_color(value, vmax, vmin, threshold):
    return "white" if (value - vmin) < (vmax - vmin) * threshold else "black"


def _plot_symmetric(rate, cmap, name, unit, xticks, xlabel, yticks, ylabel, color_threshold, vmin=None, vmax=None, baseline=None):
    fig, ax = plt.subplots()
    im = ax.imshow(rate.T, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    cbar_ax = fig.add_axes((0.85, 0.15, 0.04, 0.69))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel(name, rotation=-90, va="bottom")
    if baseline:
        cbar.ax.axhline(baseline, color="red", linestyle="--")
    vmin = np.min(rate) if vmin is None else vmin
    vmax = np.max(rate) if vmin is None else vmin
    
    ax.set_xticks(np.arange(len(xticks)), labels=xticks)
    ax.set_yticks(np.arange(len(yticks)), labels=yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for i in range(len(xticks)):
        for j in range(len(yticks)):
            value = rate[i, j]
            text = ax.text(i, j, f"{value:.1f}{unit}",
                           ha="center", va="center", color=_text_color(value, vmax, vmin, color_threshold))
    return fig


def query_rate_symmetric(query_rate):
    return _plot_symmetric(query_rate, mako, "Oracle query rate (%)", "%", errs, "error probability", majs, "majority vote", 0.5)


def success_rate_symmetric(correct_rate, baseline=None):
    return _plot_symmetric(correct_rate, viridis, "Success rate (%)", "%", errs, "error probability", majs, "majority vote", 0.8, vmin=0, vmax=100, baseline=baseline)


def amount_rate_symmetric(amount_rate):
    return _plot_symmetric(amount_rate, plasma, "Result size", "", errs, "error probability", majs, "majority vote", 0.5)


def precise_rate_symmetric(precise_rate):
    return _plot_symmetric(precise_rate, viridis, "Precision", "", errs, "error probability", majs, "majority vote", 0.8, vmin=0, vmax=100)


def success_rate_vs_query_rate_symmetric(query_rate, correct_rate):
    fig, ax = plt.subplots()
    ax.grid()
    for err in errs:
        qrs = query_rate.sel(err=err)
        crs = correct_rate.sel(err=err)
        ax.scatter(qrs, crs, label=f"error = {err}")
    ax.set_xlabel("oracle queries")
    ax.set_ylabel("success rate")
    ax.legend()
    return fig


def success_rate_vs_majority_symmetric(correct_rate):
    fig, ax = plt.subplots()
    ax.grid()
    for err in errs:
        crs = correct_rate.sel(err=err)
        ax.plot(list(majs), crs, label=f"error = {err}")
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
            query_rate_level = query_rate_b.isel(majority=level)
            im = ax.imshow(query_rate_level.T, cmap=mako, vmin=vmin, vmax=vmax, origin="lower")
            ax.set_xticks(np.arange(len(errs)), labels=errs)
            ax.set_yticks(np.arange(len(errs)), labels=errs)
            for i in range(len(errs)):
                for j in range(len(errs)):
                    q_rate = query_rate_level[i, j]
                    text = ax.text(i, j, f"{q_rate:.0f}", ha="center", va="center", color=_text_color(q_rate, vmax, vmin, 0.5))
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
            correct_rate_level = correct_rate_b.isel(majority=level)
            im = ax.imshow(correct_rate_level.T, cmap=viridis, vmin=0, vmax=100, origin="lower")
            ax.set_xticks(np.arange(len(errs)), labels=errs)
            ax.set_yticks(np.arange(len(errs)), labels=errs)
            for i in range(len(errs)):
                for j in range(len(errs)):
                    c_rate = correct_rate_level[i, j]
                    text = ax.text(i, j, f"{c_rate:.0f}%", ha="center", va="center", color=_text_color(c_rate, 100, 0, 0.5))
            ax.set_xlabel("$e_1$")
            ax.set_ylabel("$e_O$")
            ax.set_title(majs[level])
    fig.set_size_inches((10,6))
    fig.tight_layout(h_pad=1.5, rect=(0, 0, 0.9, 1))
    cbar_ax = fig.add_axes((0.9, 0.10, 0.02, 0.84))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Success rate", rotation=-90, va="bottom")
    if baseline:
        cbar.ax.axhline(baseline, color="red", linestyle="--")
    return fig


def amount_rate_asymmetric(amount_rate_b):
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex="col", sharey="row")
    vmin = np.min(amount_rate_b)
    vmax = np.max(amount_rate_b)
    
    for row in range(2):
        for col in range(3):
            ax = axs[row, col]
            level = row * 3 + col
            amount_rate_level = amount_rate_b.isel(majority=level)
            im = ax.imshow(amount_rate_level.T, cmap=plasma, vmin=vmin, vmax=vmax, origin="lower")
            ax.set_xticks(np.arange(len(errs)), labels=errs)
            ax.set_yticks(np.arange(len(errs)), labels=errs)
            for i in range(len(errs)):
                for j in range(len(errs)):
                    a_rate = amount_rate_level[i, j]
                    text = ax.text(i, j, f"{a_rate:.0f}", ha="center", va="center", color=_text_color(a_rate, vmax, vmin, 0.5))
            ax.set_xlabel("$e_1$")
            ax.set_ylabel("$e_O$")
            ax.set_title(majs[level])
    fig.set_size_inches((10,6))
    fig.tight_layout(h_pad=1.5, rect=(0, 0, 0.9, 1))
    cbar_ax = fig.add_axes((0.9, 0.10, 0.02, 0.84))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Result size", rotation=-90, va="bottom")
    return fig


def precise_rate_asymmetric(precise_rate_b):
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex="col", sharey="row")
    for row in range(2):
        for col in range(3):
            ax = axs[row, col]
            level = row * 3 + col
            precise_rate_level = precise_rate_b.isel(majority=level)
            im = ax.imshow(precise_rate_level.T, cmap=viridis, vmin=0, vmax=100, origin="lower")
            ax.set_xticks(np.arange(len(errs)), labels=errs)
            ax.set_yticks(np.arange(len(errs)), labels=errs)
            for i in range(len(errs)):
                for j in range(len(errs)):
                    p_rate = precise_rate_level[i, j]
                    text = ax.text(i, j, f"{p_rate:.0f}%", ha="center", va="center", color=_text_color(p_rate, 100, 0, 0.5))
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
    for err_0 in errs:
        for err_1 in errs:
            crs = correct_rate_b.sel(err_0=err_0, err_1=err_1)
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
    im = ax.imshow(query_rate.T, cmap=mako, origin="lower")
    cbar_ax = fig.add_axes((0.85, 0.15, 0.04, 0.69))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Oracle query rate", rotation=-90, va="bottom")

    vmin = np.min(query_rate)
    vmax = np.max(query_rate)
    
    ax.set_xticks(np.arange(len(nums)), labels=nums)
    ax.set_yticks(np.arange(len(smpls)), labels=smpls)
    ax.set_xlabel("binom n")
    ax.set_ylabel("samples")
    for i in range(len(nums)):
        for j in range(len(smpls)):
            q_rate = query_rate[i, j]
            text = ax.text(i, j, f"{q_rate:.1f}",
                           ha="center", va="center", color=_text_color(q_rate, vmax, vmin, 0.5))
    return fig


def success_rate_binomial(correct_rate, baseline):
    fig, ax = plt.subplots()
    im = ax.imshow(correct_rate.T, vmin=0, cmap=viridis, origin="lower")
    cbar_ax = fig.add_axes((0.85, 0.15, 0.04, 0.69))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Success rate", rotation=-90, va="bottom")
    if baseline:
        cbar.ax.axhline(baseline, color="red", linestyle="--")

    vmin = 0
    vmax = np.max(correct_rate)
    
    ax.set_xticks(np.arange(len(nums)), labels=nums)
    ax.set_yticks(np.arange(len(smpls)), labels=smpls)
    ax.set_xlabel("binom n")
    ax.set_ylabel("samples")
    for i in range(len(nums)):
        for j in range(len(smpls)):
            c_rate = correct_rate[i, j]
            text = ax.text(i, j, f"{c_rate:.1f}%",
                           ha="center", va="center", color=_text_color(c_rate, vmax, vmin, 0.8))
    return fig


def amount_rate_binomial(amount_rate):
    fig, ax = plt.subplots()
    im = ax.imshow(amount_rate.T, cmap=plasma, origin="lower")
    cbar_ax = fig.add_axes((0.85, 0.15, 0.04, 0.69))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Result size", rotation=-90, va="bottom")

    vmin = np.min(amount_rate)
    vmax = np.max(amount_rate)
    
    ax.set_xticks(np.arange(len(nums)), labels=nums)
    ax.set_yticks(np.arange(len(smpls)), labels=smpls)
    ax.set_xlabel("binom n")
    ax.set_ylabel("samples")
    for i in range(len(nums)):
        for j in range(len(smpls)):
            a_rate = amount_rate[i, j]
            text = ax.text(i, j, f"{a_rate:.1f}",
                           ha="center", va="center", color=_text_color(a_rate, vmax, vmin, 0.5))
    return fig


def precise_rate_binomial(precise_rate):
    fig, ax = plt.subplots()
    im = ax.imshow(precise_rate.T, vmin=0, cmap=viridis, origin="lower")
    cbar_ax = fig.add_axes((0.85, 0.15, 0.04, 0.69))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Precision", rotation=-90, va="bottom")

    vmin = 0
    vmax = np.max(precise_rate)
    
    ax.set_xticks(np.arange(len(nums)), labels=nums)
    ax.set_yticks(np.arange(len(smpls)), labels=smpls)
    ax.set_xlabel("binom n")
    ax.set_ylabel("samples")
    for i in range(len(nums)):
        for j in range(len(smpls)):
            p_rate = precise_rate[i, j]
            text = ax.text(i, j, f"{p_rate:.1f}%",
                           ha="center", va="center", color=_text_color(p_rate, vmax, vmin, 0.8))
    return fig
