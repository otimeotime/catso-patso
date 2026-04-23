#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace mcts::exp {
    enum class RewardDistribution {
        Gaussian,
        Bernoulli
    };

    /**
     * Synthetic tree environment.
     *
     * State: (depth, index) encoded as IntPairState, with root (0,0).
     * Action: integer in [0, branching_factor-1], selecting a child:
     *   (depth+1, index * branching_factor + action).
     *
     * Transition noise: with probability action_noise, the executed action is
     * sampled uniformly from all actions (including the intended one).
     *
     * Reward: zero everywhere except at leaves (depth == max_depth), where each
     * leaf has a fixed reward sampled at construction (or per rollout if enabled).
     * If reward_mean_values /
     * reward_std_values are provided, they are used per-leaf in breadth-first
     * leaf index order (shorter arrays repeat the last value, longer arrays
     * are truncated). Otherwise, a single mean/std default is used for all leaves.
     * For Bernoulli, the mean array is used as p in [0,1] and the std array is
     * ignored. Reward is discounted by gamma^depth.
     */
    class SyntheticTreeEnvContext : public mcts::MctsEnvContext {
        public:
            std::vector<double> leaf_rewards;
    };

    class SyntheticTreeEnv : public mcts::MctsEnv {
        int branching_factor;
        int max_depth;
        double gamma;
        double action_noise;
        double reward_mean_default;
        double reward_std_default;
        RewardDistribution reward_dist;
        std::vector<double> reward_mean_values;
        std::vector<double> reward_std_values;
        std::vector<int> nodes_per_depth;
        std::vector<double> leaf_rewards;
        int reward_seed;
        bool resample_rewards_each_rollout;
        mutable std::atomic<uint64_t> rollout_counter;
        static thread_local const SyntheticTreeEnvContext* tls_ctx;

        int get_num_nodes_at_depth(int depth) const;
        std::shared_ptr<const mcts::IntPairState> make_child_state(
            std::shared_ptr<const mcts::IntPairState> state,
            int action) const;
        double reward_for_state(std::shared_ptr<const mcts::IntPairState> state) const;
        void init_leaf_rewards(int seed);
        void build_leaf_rewards(std::mt19937& rng, std::vector<double>& out) const;

        public:
            SyntheticTreeEnv(
                int branching_factor,
                int max_depth,
                double gamma = 1.0,
                double action_noise = 0.0,
                int reward_seed = 12345,
                RewardDistribution reward_dist = RewardDistribution::Gaussian,
                std::vector<double> reward_mean_values = {},
                std::vector<double> reward_std_values = {},
                double reward_mean_default = 0.0,
                double reward_std_default = 1.0,
                bool resample_rewards_each_rollout = true);
            virtual ~SyntheticTreeEnv() = default;

            // Typed API
            std::shared_ptr<const mcts::IntPairState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const mcts::IntPairState> state) const;
            std::shared_ptr<mcts::IntActionVector> get_valid_actions(
                std::shared_ptr<const mcts::IntPairState> state) const;
            std::shared_ptr<mcts::IntPairStateDistr> get_transition_distribution(
                std::shared_ptr<const mcts::IntPairState> state,
                std::shared_ptr<const mcts::IntAction> action) const;
            std::shared_ptr<const mcts::IntPairState> sample_transition_distribution(
                std::shared_ptr<const mcts::IntPairState> state,
                std::shared_ptr<const mcts::IntAction> action,
                mcts::RandManager& rand_manager) const;
            double get_reward(
                std::shared_ptr<const mcts::IntPairState> state,
                std::shared_ptr<const mcts::IntAction> action,
                std::shared_ptr<const mcts::IntPairState> observation = nullptr) const;

            // Interface API
            virtual std::shared_ptr<const mcts::State> get_initial_state_itfc() const;
            virtual bool is_sink_state_itfc(std::shared_ptr<const mcts::State> state) const;
            virtual std::shared_ptr<mcts::ActionVector> get_valid_actions_itfc(
                std::shared_ptr<const mcts::State> state) const;
            virtual std::shared_ptr<mcts::StateDistr> get_transition_distribution_itfc(
                std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action) const;
            virtual std::shared_ptr<const mcts::State> sample_transition_distribution_itfc(
                std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action,
                mcts::RandManager& rand_manager) const;
            virtual double get_reward_itfc(
                std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action,
                std::shared_ptr<const mcts::Observation> observation = nullptr) const;
            virtual std::shared_ptr<mcts::MctsEnvContext> sample_context_itfc(
                std::shared_ptr<const mcts::State> state) const;

            int get_branching_factor() const { return branching_factor; }
            int get_max_depth() const { return max_depth; }
            double get_gamma() const { return gamma; }
            double get_action_noise() const { return action_noise; }
            int get_num_leaves() const { return nodes_per_depth.empty() ? 0 : nodes_per_depth.back(); }
            RewardDistribution get_reward_distribution() const { return reward_dist; }
    };
}
