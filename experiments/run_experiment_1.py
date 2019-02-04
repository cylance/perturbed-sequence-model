from experiments.util import set_matplotlib_engine, ensure_dir

set_matplotlib_engine()
from experiments.consistency.empirical_consistency import (
    make_empirical_consistency_heat_map,
)


def run_experiment_1():
    ensure_dir("plots")
    make_empirical_consistency_heat_map()


if __name__ == "__main__":
    run_experiment_1()
