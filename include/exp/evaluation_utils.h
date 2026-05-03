#pragma once

#include "mcts_decision_node.h"
#include "mcts_env.h"

#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace mcts::exp {
    inline constexpr double kCvarTolerance = 1e-12;

    struct EvalStats {
        double mean = 0.0;
        double stddev = 0.0;
        double cvar = 0.0;
        double cvar_stddev = 0.0;
        int catastrophic_count = 0;
    };

    struct TailStats {
        double mean = 0.0;
        double stddev = 0.0;
    };

    using ReturnDistribution = std::map<double, double>;

    double compute_lower_tail_cvar(const ReturnDistribution& distribution, double tau);

    TailStats compute_empirical_lower_tail_stats(std::vector<double> samples, double tau);

    EvalStats evaluate_tree_generic(
        std::shared_ptr<const mcts::MctsEnv> env,
        std::shared_ptr<const mcts::MctsDNode> root,
        int horizon,
        int rollouts,
        int threads,
        int seed,
        double cvar_tau,
        const std::function<bool(
            std::shared_ptr<const mcts::State>,
            std::shared_ptr<const mcts::Action>,
            std::shared_ptr<const mcts::State>)>& is_catastrophic_transition = {});
}
