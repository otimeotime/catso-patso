#pragma once

#include "exp/discrete_runner_common.h"
#include "exp/experiment_spec.h"
#include "exp/oracles/discrete_cvar_oracle_common.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace mcts::exp::env_specs {
    inline void disable_all_run_algorithms(RunCandidateConfig& config) {
        config.uct.enabled = false;
        config.max_uct.enabled = false;
        config.power_uct.enabled = false;
        config.catso.enabled = false;
        config.patso.enabled = false;
    }

    inline void disable_all_tuning_algorithms(TuningGridConfig& config) {
        config.uct.enabled = false;
        config.max_uct.enabled = false;
        config.power_uct.enabled = false;
        config.catso.enabled = false;
        config.patso.enabled = false;
    }

    inline void configure_uct_family_run_config(
        RunCandidateConfig& config,
        double epsilon,
        double bias = mcts::UctManagerArgs::USE_AUTO_BIAS,
        bool use_auto_bias = true,
        double power_mean_exponent = 1.0)
    {
        config.uct.enabled = true;
        config.uct.bias = bias;
        config.uct.use_auto_bias = use_auto_bias;
        config.uct.epsilon = epsilon;

        config.max_uct.enabled = true;
        config.max_uct.bias = bias;
        config.max_uct.use_auto_bias = use_auto_bias;
        config.max_uct.epsilon = epsilon;

        config.power_uct.enabled = true;
        config.power_uct.bias = bias;
        config.power_uct.use_auto_bias = use_auto_bias;
        config.power_uct.epsilon = epsilon;
        config.power_uct.power_mean_exponent = power_mean_exponent;
    }

    inline void configure_distributional_run_config(
        RunCandidateConfig& config,
        int catso_n_atoms,
        double catso_optimism,
        int patso_max_particles,
        double patso_optimism,
        double power_mean_exponent = 1.0)
    {
        config.catso.enabled = true;
        config.catso.n_atoms = catso_n_atoms;
        config.catso.optimism = catso_optimism;
        config.catso.power_mean_exponent = power_mean_exponent;

        config.patso.enabled = true;
        config.patso.max_particles = patso_max_particles;
        config.patso.optimism = patso_optimism;
        config.patso.power_mean_exponent = power_mean_exponent;
    }

    inline void configure_uct_family_tuning_grid(
        TuningGridConfig& config,
        std::vector<double> bias_values,
        std::vector<double> epsilon_values,
        std::vector<double> power_mean_exponent_values)
    {
        config.uct.enabled = true;
        config.uct.bias_values = bias_values;
        config.uct.epsilon_values = epsilon_values;

        config.max_uct.enabled = true;
        config.max_uct.bias_values = bias_values;
        config.max_uct.epsilon_values = epsilon_values;

        config.power_uct.enabled = true;
        config.power_uct.bias_values = std::move(bias_values);
        config.power_uct.epsilon_values = std::move(epsilon_values);
        config.power_uct.power_mean_exponent_values = std::move(power_mean_exponent_values);
    }

    inline void configure_distributional_tuning_grid(
        TuningGridConfig& config,
        std::vector<int> catso_n_atoms_values,
        std::vector<double> catso_optimism_values,
        std::vector<int> patso_max_particles_values,
        std::vector<double> patso_optimism_values,
        std::vector<double> power_mean_exponent_values,
        std::vector<double> tau_values)
    {
        config.catso.enabled = true;
        config.catso.n_atoms_values = std::move(catso_n_atoms_values);
        config.catso.optimism_values = std::move(catso_optimism_values);
        config.catso.power_mean_exponent_values = power_mean_exponent_values;
        config.catso.tau_values = tau_values;

        config.patso.enabled = true;
        config.patso.max_particles_values = std::move(patso_max_particles_values);
        config.patso.optimism_values = std::move(patso_optimism_values);
        config.patso.power_mean_exponent_values = std::move(power_mean_exponent_values);
        config.patso.tau_values = std::move(tau_values);
    }

    inline void apply_generic_tuning_defaults(ExperimentSpec& spec) {
        if (!spec.trial_counts.empty()) {
            spec.tune_total_trials = spec.trial_counts.back();
        }
        spec.tune_runs = std::max(1, std::min(spec.runs, 3));
        spec.tune_threads = spec.threads;
        spec.progress_batch_trials = std::max(1, std::min(5000, spec.tune_total_trials));
    }

    template <typename EnvT, typename StateT>
    inline std::function<RootMetrics(
        std::shared_ptr<const mcts::MctsEnv>,
        std::shared_ptr<const mcts::MctsDNode>,
        double,
        double
    )> make_generic_root_metrics_evaluator(double optimal_action_regret_threshold = 0.0)
    {
        auto oracle_cache =
            std::make_shared<std::map<
                std::pair<double, double>,
                std::shared_ptr<mcts::exp::runner::CvarOracle<EnvT, StateT>>>>();

        return [oracle_cache, optimal_action_regret_threshold](
            std::shared_ptr<const mcts::MctsEnv> generic_env,
            std::shared_ptr<const mcts::MctsDNode> root,
            double eval_tau,
            double discount_gamma)
        {
            RootMetrics metrics;

            const auto env = std::static_pointer_cast<const EnvT>(generic_env);
            auto& oracle = (*oracle_cache)[{eval_tau, discount_gamma}];
            if (!oracle) {
                oracle = std::make_shared<mcts::exp::runner::CvarOracle<EnvT, StateT>>(
                    env,
                    eval_tau,
                    discount_gamma);
            }

            const auto& root_solution = oracle->solve_state(env->get_initial_state());
            auto context = env->sample_context_itfc(env->get_initial_state_itfc());
            std::shared_ptr<const mcts::Action> recommended_action = root->recommend_action_itfc(*context);
            const int recommended_action_id =
                std::static_pointer_cast<const mcts::IntAction>(recommended_action)->action;

            const auto action_cvar_it = root_solution.action_cvars.find(recommended_action_id);
            if (action_cvar_it == root_solution.action_cvars.end()) {
                return metrics;
            }

            const double estimated_action_cvar =
                mcts::exp::oracles::estimate_recommended_action_cvar(root, recommended_action, eval_tau);
            if (std::isnan(estimated_action_cvar)) {
                return metrics;
            }

            metrics.cvar_regret = std::abs(root_solution.optimal_cvar - estimated_action_cvar);
            metrics.optimal_action_hit =
                (metrics.cvar_regret <= optimal_action_regret_threshold + mcts::exp::kCvarTolerance) ? 1.0 : 0.0;
            return metrics;
        };
    }
}
