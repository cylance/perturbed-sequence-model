from experiments.util import set_matplotlib_engine, ensure_dir

set_matplotlib_engine()

from experiments.configs import model_path_props, model_path_rnn, model_path_hmm
from experiments.trump.get_trump_tweets import get_sample_of_trump_tweets
from experiments.trump.get_trained_model import get_trained_model
from experiments.trump.figures_for_paper.collocations_illustration import (
    make_collocations_illustration,
)
from experiments.trump.figures_for_paper.customization_illustration import (
    make_customization_illustration,
)
from experiments.trump.figures_for_paper.lecunn_illustration import (
    make_lecunn_illustration,
)


def run_experiment_2():

    ensure_dir("plots/")

    ### ----- Data -----
    sample = get_sample_of_trump_tweets()

    ### ---- Training-----
    # ( Train the models on the trump tweets.)
    # Note that we provide the models in the repo.
    # However, if you delete them from locations model_path_rnn and model_path_rnn (specified in experiments/configs.py),
    # then you can create new models.
    props_model = get_trained_model(
        sample, model_path_props, model_path_rnn, model_type="props", K=3, n_train=2000
    )
    hmm_model = get_trained_model(
        sample, model_path_hmm, model_path_rnn, model_type="hmm", K=2, n_train=2000
    )

    ### ---- Scoring and Plotting ----
    # (Produce the plots in the paper.)
    make_collocations_illustration(props_model, hmm_model, model_path_rnn, sample)
    make_customization_illustration(props_model, model_path_rnn, sample)
    make_lecunn_illustration(props_model, model_path_rnn)


if __name__ == "__main__":
    run_experiment_2()
