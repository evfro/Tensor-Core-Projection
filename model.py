import numpy as np
from hooi import sparse_hooi
from dataprep import user_field, item_field, position_field


def seqtf_model_build(config, data, data_description, iter_callback=None, verbose=True):
    users = data_description[user_field]
    items = data_description[item_field]
    positions = data_description[position_field]

    n_users = data_description["n_users"]
    n_items = data_description["n_items"]
    max_pos = data_description["n_pos"]
    shape = (n_users, n_items, max_pos)

    idx = data[[users, items, positions]].values
    val = np.ones(idx.shape[0], dtype='f8')

    return sparse_hooi(
        idx, val, shape, config["mlrank"],
        max_iters = config["max_iters"],
        update_order=config["update_order"],
        materialize_core=False,
        growth_tol = config["growth_tol"],
        seed = config["seed"],
        iter_callback=iter_callback,
        verbose=verbose,
    )


def tf_scoring(params, data, data_description, core_projected=True):
    user_factors, item_factors, pos_factors, core = params
    if core_projected:
        assert isinstance(core, (tuple, list))
        assert len(core[0]) == user_factors.shape[1]
    else:
        core = None

    userid = data_description[user_field]
    itemid = data_description[item_field]
    posid = data_description[position_field]

    tset_data = data.sort_values([userid, posid])
    useridx = tset_data[userid].values
    itemidx = tset_data[itemid].values
    indptr, = np.where(np.diff(useridx, prepend=-1, append=-1))
    scores = user_scoring(indptr, itemidx, item_factors, pos_factors, core)
    return scores

def user_scoring(indptr, indices, item_factors, pos_factors, core):
    sequences = np.array_split(indices, indptr[1:-1])
    n_items = item_factors.shape[0]
    scores = np.zeros((len(sequences), n_items))
    for u, seq in enumerate(sequences):
        scores[u] = sequences_score(seq, item_factors, pos_factors, core)
    return scores

def sequences_score(seq, item_factors, pos_factors, core):
    n_pos, _ = pos_factors.shape
    user_profile = item_factors[seq[-(n_pos-1):], :]
    seq_length, _ = user_profile.shape
    compressed = user_profile.T @ pos_factors[-(seq_length+1):-1]
    if core is not None:
        singular_values, core_factors = core
        nnz_sv = singular_values > np.finfo(core_factors.dtype).eps
        if not nnz_sv.all():
            core_factors = core_factors[nnz_sv, :]
        compressed = np.reshape(
            core_factors.T @ (core_factors @ compressed.ravel(order='F')),
            compressed.shape,
            order='F'
        )
    scores = item_factors @ (compressed @ pos_factors[-1, :])
    return scores