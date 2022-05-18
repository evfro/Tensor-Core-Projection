from polara import get_movielens_data
from polara.lib.earlystopping import early_stopping_callback
from polara.evaluation.pipelines import random_grid
from polara.tools.display import suppress_stdout

from dataprep import prepare_sequential_data
from dataprep import user_field, item_field
from experiment import TFParamStore, tf_evaluator
from model import seqtf_model_build
from hooi import valid_mlrank

n_pos = 200
time_q = 0.90
data_path = 'ml-1m.zip'
max_fails = 2


if __name__ == "__main__":
    mldata = get_movielens_data(data_path, include_time=True)
    with suppress_stdout():
        train_pack, valid_pack, test_pack = prepare_sequential_data(mldata, n_pos, time_q)
    training_data, data_index = train_pack
    data_description = dict(
        userid = data_index[user_field].name,
        itemid = data_index[item_field].name,
        positionid = 'pos',
        n_users = len(data_index[user_field]),
        n_items = len(data_index[item_field]),
        n_pos = n_pos
    )

    common_config = {
        "n_pos": n_pos,
        "max_iters": 15,
        "growth_tol": 1e-6,
        "seed": 99,
    }

    grid = dict(
        r_user = [24, 32, 48, 64, 96, 128, 192],
        r_item = [24, 32, 48, 64, 96, 128, 192],
        r_pos = [4, 8, 16, 24],
        projected = [True, False],
        update_order = [(2, 1, 0), (1, 2, 0), (0, 1, 2), (1, 0, 2), (2, 0, 1), (0, 2, 1)]
    )

    def skip_config(config):
        *mlrank, projected, update_order = config
        if not valid_mlrank(mlrank):
            return True
        if projected and update_order[-1] != 0:
            return True    
    
    res = {}
    res['original'] = {}
    res['coreless'] = {}
    res['projected'] = {}
    param_grid, param_names = random_grid(grid, n=0, skip_config=skip_config)

    with open(f'{n_pos=}_{time_q=}_{data_path=:s}_{max_fails=}.csv', 'w') as res_file:
        res_file.write(f'label,HR,{",".join(param_names):s},iters\n')
        for *mlrank, projected, update_order in param_grid:
            config = {
                "mlrank": tuple(mlrank),
                "projected": projected,
                "update_order": update_order,
            }
            optimal_params = TFParamStore()
            evaluator = tf_evaluator(
                *valid_pack, data_description, core_projected=config.pop("projected")
            )
            early_stopper = early_stopping_callback(
                evaluator, max_fails=max_fails, param_store=optimal_params, verbose=False
            )
            model_params = seqtf_model_build(
                {**common_config, **config},
                training_data,
                data_description,
                iter_callback=early_stopper,
                verbose=False
            )

            if projected:
                label = 'projected'
            elif update_order[-1] == 0:
                label = 'coreless' # same order as projected
            else:
                label = 'original'
            current_res = early_stopper.target
            num_iters = early_stopper.iter + 1
            res_file.write(
                f'{label},{current_res},{",".join(map(str, mlrank)):s},{projected},"{update_order}",{num_iters}\n'
            )
            res_file.flush()
            if current_res > max(res[label].values(), default=float('-inf')):
                print(f'New best {label} model result HR={current_res:.4f} with {config}')
            res[label][tuple(config.values())] = current_res


    