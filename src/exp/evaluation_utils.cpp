#include "exp/evaluation_utils.h"

#include "mc_eval.h"
#include "mcts.h"
#include "mcts_env_context.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace mcts::exp {
    double compute_lower_tail_cvar(const ReturnDistribution& distribution, double tau) {
        if (distribution.empty()) {
            return 0.0;
        }

        const double tau_clamped = std::max(kCvarTolerance, std::min(tau, 1.0));
        double total_mass = 0.0;
        for (const auto& [value, probability] : distribution) {
            (void)value;
            if (probability > 0.0) {
                total_mass += probability;
            }
        }

        if (total_mass <= 0.0) {
            return 0.0;
        }

        const double tail_mass = tau_clamped * total_mass;
        double cumulative_mass = 0.0;
        double tail_sum = 0.0;

        for (const auto& [value, probability] : distribution) {
            if (probability <= 0.0) {
                continue;
            }

            const double remaining_mass = std::max(0.0, tail_mass - cumulative_mass);
            const double mass_to_take = std::min(probability, remaining_mass);
            tail_sum += mass_to_take * value;
            cumulative_mass += probability;

            if (cumulative_mass >= tail_mass) {
                break;
            }
        }

        return tail_sum / tail_mass;
    }

    TailStats compute_empirical_lower_tail_stats(std::vector<double> samples, double tau) {
        TailStats stats;
        if (samples.empty()) {
            return stats;
        }

        std::sort(samples.begin(), samples.end());

        const double tau_clamped = std::max(kCvarTolerance, std::min(tau, 1.0));
        const double tail_mass = tau_clamped * static_cast<double>(samples.size());
        double cumulative_mass = 0.0;
        double tail_sum = 0.0;
        double weighted_square_sum = 0.0;

        for (double sample : samples) {
            const double remaining_mass = std::max(0.0, tail_mass - cumulative_mass);
            const double mass_to_take = std::min(1.0, remaining_mass);
            if (mass_to_take <= 0.0) {
                break;
            }

            tail_sum += mass_to_take * sample;
            weighted_square_sum += mass_to_take * sample * sample;
            cumulative_mass += mass_to_take;

            if (cumulative_mass >= tail_mass) {
                break;
            }
        }

        if (cumulative_mass <= 0.0) {
            return stats;
        }

        stats.mean = tail_sum / cumulative_mass;
        const double second_moment = weighted_square_sum / cumulative_mass;
        stats.stddev = std::sqrt(std::max(0.0, second_moment - stats.mean * stats.mean));
        return stats;
    }

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
            std::shared_ptr<const mcts::State>)>& is_catastrophic_transition)
    {
        (void)threads;

        mcts::RandManager eval_rng(seed);
        mcts::EvalPolicy policy(root, env, eval_rng);
        std::vector<double> sampled_returns;
        sampled_returns.reserve(static_cast<size_t>(std::max(rollouts, 0)));
        int catastrophic_count = 0;
        const bool debug_trajectories = mcts::should_debug_mc_eval_trajectories_from_env();
        const std::string trajectory_label = "MC";

        for (int rollout = 0; rollout < rollouts; ++rollout) {
            policy.reset();

            int num_actions_taken = 0;
            double sample_return = 0.0;
            std::shared_ptr<const mcts::State> state = env->get_initial_state_itfc();
            bool rollout_was_catastrophic = env->is_catastrophic_state_itfc(state);
            auto context_ptr = env->sample_context_itfc(state);
            mcts::MctsEnvContext& context = *context_ptr;
            mcts::RolloutTrajectoryTrace trajectory_trace(debug_trajectories, trajectory_label);
            trajectory_trace.append_state(state);

            while (num_actions_taken < horizon && !env->is_sink_state_itfc(state)) {
                std::shared_ptr<const mcts::Action> action = policy.get_action(state, context);
                std::shared_ptr<const mcts::State> next_state =
                    env->sample_transition_distribution_itfc(state, action, eval_rng);
                std::shared_ptr<const mcts::Observation> observation =
                    env->sample_observation_distribution_itfc(action, next_state, eval_rng);

                const double reward = env->get_reward_itfc(state, action, observation);
                sample_return += reward;
                trajectory_trace.append_reward(reward);
                trajectory_trace.append_state(next_state);
                if ((is_catastrophic_transition && is_catastrophic_transition(state, action, next_state))
                    || (!is_catastrophic_transition && env->is_catastrophic_state_itfc(next_state))) {
                    rollout_was_catastrophic = true;
                }

                policy.update_step(action, observation);
                state = next_state;
                ++num_actions_taken;
            }

            trajectory_trace.print(std::cerr);
            sampled_returns.push_back(sample_return);
            if (rollout_was_catastrophic) {
                ++catastrophic_count;
            }
        }

        if (sampled_returns.empty()) {
            return {};
        }

        EvalStats stats;
        for (double sample_return : sampled_returns) {
            stats.mean += sample_return;
        }
        stats.mean /= static_cast<double>(sampled_returns.size());

        if (sampled_returns.size() > 1u) {
            double variance = 0.0;
            for (double sample_return : sampled_returns) {
                variance += std::pow(sample_return - stats.mean, 2.0);
            }
            variance /= static_cast<double>(sampled_returns.size() - 1u);
            stats.stddev = std::sqrt(variance);
        }

        const TailStats tail_stats = compute_empirical_lower_tail_stats(sampled_returns, cvar_tau);
        stats.cvar = tail_stats.mean;
        stats.cvar_stddev = tail_stats.stddev;
        stats.catastrophic_count = catastrophic_count;
        return stats;
    }
}
