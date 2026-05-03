#pragma once

#include "algorithms/catso/catso_chance_node.h"
#include "algorithms/catso/catso_decision_node.h"
#include "algorithms/uct/max_uct_chance_node.h"
#include "algorithms/uct/max_uct_decision_node.h"
#include "algorithms/uct/power_uct_chance_node.h"
#include "algorithms/uct/power_uct_decision_node.h"
#include "algorithms/uct/uct_chance_node.h"
#include "algorithms/uct/uct_decision_node.h"
#include "exp/evaluation_utils.h"
#include "exp/experiment_spec.h"
#include "mcts_env.h"
#include "mcts_types.h"

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace mcts::exp::oracles {
    struct OptimalDistributionSolution {
        ReturnDistribution state_value_distribution;
        std::map<int, ReturnDistribution> action_value_distributions;
        std::map<int, double> action_cvars;
        std::vector<int> optimal_actions;
        int canonical_action = -1;
        double optimal_cvar = 0.0;
    };

    inline double estimate_recommended_action_cvar(
        std::shared_ptr<const mcts::MctsDNode> root,
        std::shared_ptr<const mcts::Action> recommended_action,
        double eval_tau)
    {
        if (const auto catso_root = std::dynamic_pointer_cast<const mcts::CatsoDNode>(root)) {
            if (catso_root->has_child_node(recommended_action)) {
                return catso_root->get_child_node(recommended_action)->get_cvar_value_at(eval_tau);
            }
        }

        if (const auto uct_root = std::dynamic_pointer_cast<const mcts::UctDNode>(root)) {
            if (uct_root->has_child_node(recommended_action)) {
                return uct_root->get_child_node(recommended_action)->get_cvar_value_at(eval_tau);
            }
        }

        if (const auto max_uct_root = std::dynamic_pointer_cast<const mcts::MaxUctDNode>(root)) {
            if (max_uct_root->has_child_node(recommended_action)) {
                return max_uct_root->get_child_node(recommended_action)->get_cvar_value_at(eval_tau);
            }
        }

        if (const auto power_uct_root = std::dynamic_pointer_cast<const mcts::PowerUctDNode>(root)) {
            if (power_uct_root->has_child_node(recommended_action)) {
                return power_uct_root->get_child_node(recommended_action)->get_cvar_value_at(eval_tau);
            }
        }

        return std::numeric_limits<double>::quiet_NaN();
    }

    inline RootMetrics evaluate_root_action_cvar_metrics(
        std::shared_ptr<const mcts::MctsEnv> env,
        std::shared_ptr<const mcts::MctsDNode> root,
        const OptimalDistributionSolution& root_solution,
        double eval_tau,
        double optimal_action_regret_threshold = 0.0)
    {
        RootMetrics metrics;

        auto context = env->sample_context_itfc(env->get_initial_state_itfc());
        std::shared_ptr<const mcts::Action> recommended_action = root->recommend_action_itfc(*context);
        const int recommended_action_id =
            std::static_pointer_cast<const mcts::IntAction>(recommended_action)->action;

        const auto action_cvar_it = root_solution.action_cvars.find(recommended_action_id);
        if (action_cvar_it == root_solution.action_cvars.end()) {
            return metrics;
        }

        const double estimated_action_cvar = estimate_recommended_action_cvar(root, recommended_action, eval_tau);
        if (std::isnan(estimated_action_cvar)) {
            return metrics;
        }

        metrics.cvar_regret = root_solution.optimal_cvar - estimated_action_cvar;
        metrics.optimal_action_hit =
            (metrics.cvar_regret <= optimal_action_regret_threshold + kCvarTolerance) ? 1.0 : 0.0;
        return metrics;
    }
}
