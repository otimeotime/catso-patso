#pragma once

#include "exp/algorithm_factory.h"
#include "exp/experiment_spec.h"

#include <vector>

namespace mcts::exp {
    int run_tuning_experiment(
        const ExperimentSpec& spec,
        const std::vector<Candidate>& candidates);
}
