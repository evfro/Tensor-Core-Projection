from model import tf_scoring
from evaluation import downvote_seen_items, model_evaluate, topn_recommendations


def tf_evaluator(testset, holdout, data_description, core_projected, topn=10):
    def iter_evaluate(core_factors, factors):
        model_params = tuple(factors) + (core_factors,)
        scores = tf_scoring(model_params, testset, data_description, core_projected=core_projected) 
        downvote_seen_items(scores, testset, data_description)
        top_recs = topn_recommendations(scores, topn=topn)
        hr, *_ = model_evaluate(top_recs, holdout, data_description)
        return hr
    return iter_evaluate


class TFParamStore:
    core = None
    factors = None
    def __call__(self, core_factors, factors):
        self.core = core_factors
        self.factors = factors