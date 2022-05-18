import numpy as np
from polara.preprocessing.dataframes import leave_one_out, reindex


user_field = 'userid'
item_field = 'itemid'
position_field = 'positionid'


def transform_indices(data, userid, itemid):
    data_index = {}
    for entity, field in zip([user_field, item_field], [userid, itemid]):
        idx, idx_map = to_numeric_id(data, field)
        data_index[entity] = idx_map
        data.loc[:, field] = idx
    return data, data_index


def to_numeric_id(data, field):
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map


def split_data(data, userid='userid', itemid='movieid', timeid='timestamp', time_q=0.95, seed=None):
    timepoint = data[timeid].quantile(q=time_q, interpolation='nearest')
    test_data_ = data.query(f'{timeid} >= @timepoint')
    warm_users = test_data_[userid].unique()
    train_data_ = data.query(f'{userid} not in @warm_users and {timeid} < @timepoint')
    training, data_index = transform_indices(train_data_.copy(), userid, itemid)
    
    test_data = reindex(test_data_, data_index[item_field])
    random_state = None if seed is None else np.random.RandomState(seed)
    leave_one_out_config = dict(target=timeid, sample_top=True, random_state=random_state)
    # final test data
    testset_, holdout_ = leave_one_out(test_data, **leave_one_out_config)
    testset, holdout = align_test_data(testset_, holdout_)
    # validation data
    testset_valid_, holdout_valid_ = leave_one_out(testset_, **leave_one_out_config)
    testset_valid, holdout_valid = align_test_data(testset_valid_, holdout_valid_)
    return (training, data_index), (testset_valid, holdout_valid), (testset, holdout) 


def align_test_data(testset, holdout, userid='userid'):
    test_users = np.intersect1d(testset.userid.unique(), holdout.userid.unique())
    testset = testset.query(f'{userid} in @test_users').sort_values(userid)
    holdout = holdout.query(f'{userid} in @test_users').sort_values(userid)
    return testset, holdout


def assign_positions(data, maxlen, userid='userid', itemid='movieid', timeid='timestamp'):
    return (
        data
        .sort_values(timeid)
        .assign(pos = lambda df:
            df.groupby(userid)[itemid].transform(enumerate_events, maxlen)
        )
        .query('pos>=0')
        .sort_values([userid, timeid])
    )

def enumerate_events(s, maxlen=None):
    if maxlen is None:
        maxlen = len(s)
    return np.arange(maxlen-len(s), maxlen)


def prepare_sequential_data(data, n_pos, time_q, userid='userid', itemid='movieid', timeid='timestamp', seed=None):
    train_pack, valid_pack, test_pack = split_data(
        data, userid=userid, itemid=itemid, timeid=timeid, time_q=time_q, seed=seed
    )
    training = assign_positions(train_pack[0], n_pos)
    testset_valid = assign_positions(valid_pack[0], n_pos)
    testset = assign_positions(test_pack[0], n_pos)
    
    train_pack = (training,) + train_pack[1:]
    valid_pack = (testset_valid,) + valid_pack[1:]
    test_pack = (testset,) + test_pack[1:]
    return train_pack, valid_pack, test_pack