#include "env/synthetic_tree_env.h"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

#include "helper_templates.h"

using namespace std;

namespace mcts::exp {
    thread_local const SyntheticTreeEnvContext* SyntheticTreeEnv::tls_ctx = nullptr;

    SyntheticTreeEnv::SyntheticTreeEnv(
        int branching_factor,
        int max_depth,
        double gamma,
        double action_noise,
        int reward_seed,
        RewardDistribution reward_dist,
        std::vector<double> reward_mean_values,
        std::vector<double> reward_std_values,
        double reward_mean_default,
        double reward_std_default,
        bool resample_rewards_each_rollout)
        : mcts::MctsEnv(true),
          branching_factor(branching_factor),
          max_depth(max_depth),
          gamma(gamma),
          action_noise(action_noise),
          reward_mean_default(reward_mean_default),
          reward_std_default(reward_std_default),
          reward_dist(reward_dist),
          reward_mean_values(std::move(reward_mean_values)),
          reward_std_values(std::move(reward_std_values)),
          nodes_per_depth(),
          leaf_rewards(),
          reward_seed(reward_seed),
          resample_rewards_each_rollout(resample_rewards_each_rollout),
          rollout_counter(0)
    {
        if (branching_factor <= 0) throw runtime_error("SyntheticTreeEnv: branching_factor must be > 0");
        if (max_depth < 0) throw runtime_error("SyntheticTreeEnv: max_depth must be >= 0");
        if (gamma <= 0.0 || gamma > 1.0) throw runtime_error("SyntheticTreeEnv: gamma must be in (0,1]");
        if (action_noise < 0.0 || action_noise > 1.0) {
            throw runtime_error("SyntheticTreeEnv: action_noise must be in [0,1]");
        }
        if (reward_dist == RewardDistribution::Bernoulli) {
            if (reward_mean_default < 0.0 || reward_mean_default > 1.0) {
                throw runtime_error("SyntheticTreeEnv: Bernoulli mean default must be within [0,1]");
            }
        }
        if (reward_std_default < 0.0) {
            throw runtime_error("SyntheticTreeEnv: reward_std default must be >= 0");
        }

        nodes_per_depth.resize(max_depth + 1);
        size_t level_size = 1;
        for (int d = 0; d <= max_depth; ++d) {
            if (level_size > static_cast<size_t>(numeric_limits<int>::max())) {
                throw runtime_error("SyntheticTreeEnv: tree too large (node count exceeds int)");
            }
            nodes_per_depth[d] = static_cast<int>(level_size);
            if (d < max_depth) {
                if (level_size > numeric_limits<size_t>::max() / static_cast<size_t>(branching_factor)) {
                    throw runtime_error("SyntheticTreeEnv: tree too large (overflow)");
                }
                level_size *= static_cast<size_t>(branching_factor);
            }
        }

        init_leaf_rewards(reward_seed);
    }

    static double pick_array_value(const std::vector<double>& values, int idx, double fallback) {
        if (values.empty()) return fallback;
        if (idx >= 0 && idx < static_cast<int>(values.size())) return values[idx];
        return values.back();
    }

    void SyntheticTreeEnv::init_leaf_rewards(int seed) {
        std::mt19937 rng(static_cast<uint32_t>(seed));
        build_leaf_rewards(rng, leaf_rewards);
    }

    void SyntheticTreeEnv::build_leaf_rewards(std::mt19937& rng, std::vector<double>& out) const {
        const int num_leaves = get_num_leaves();
        out.resize(num_leaves, reward_mean_default);
        if (num_leaves == 0) return;

        for (int i = 0; i < num_leaves; ++i) {
            double mean = reward_mean_default;
            if (!reward_mean_values.empty()) {
                mean = pick_array_value(reward_mean_values, i, reward_mean_default);
            }

            if (reward_dist == RewardDistribution::Bernoulli) {
                if (mean < 0.0 || mean > 1.0) {
                    throw runtime_error("SyntheticTreeEnv: Bernoulli mean out of [0,1]");
                }
                std::bernoulli_distribution bern(mean);
                out[i] = bern(rng) ? 1.0 : 0.0;
            } else {
                double stddev = reward_std_default;
                if (!reward_std_values.empty()) {
                    stddev = pick_array_value(reward_std_values, i, reward_std_default);
                }
                if (stddev < 0.0) {
                    throw runtime_error("SyntheticTreeEnv: reward_std must be >= 0");
                }
                if (stddev == 0.0) {
                    out[i] = mean;
                } else {
                    std::normal_distribution<double> dist(mean, stddev);
                    out[i] = dist(rng);
                }
            }
        }
    }

    int SyntheticTreeEnv::get_num_nodes_at_depth(int depth) const {
        if (depth < 0 || depth > max_depth) {
            throw runtime_error("SyntheticTreeEnv: depth out of range");
        }
        return nodes_per_depth[depth];
    }

    shared_ptr<const mcts::IntPairState> SyntheticTreeEnv::get_initial_state() const {
        return make_shared<mcts::IntPairState>(0, 0);
    }

    bool SyntheticTreeEnv::is_sink_state(shared_ptr<const mcts::IntPairState> state) const {
        return state->state.first >= max_depth;
    }

    shared_ptr<mcts::IntActionVector> SyntheticTreeEnv::get_valid_actions(
        shared_ptr<const mcts::IntPairState> state) const
    {
        auto valid_actions = make_shared<mcts::IntActionVector>();
        if (is_sink_state(state)) return valid_actions;
        for (int a = 0; a < branching_factor; ++a) {
            valid_actions->push_back(make_shared<const mcts::IntAction>(a));
        }
        return valid_actions;
    }

    shared_ptr<const mcts::IntPairState> SyntheticTreeEnv::make_child_state(
        shared_ptr<const mcts::IntPairState> state,
        int action) const
    {
        if (action < 0 || action >= branching_factor) {
            throw runtime_error("SyntheticTreeEnv: invalid action index");
        }
        const int depth = state->state.first;
        const int index = state->state.second;
        if (depth < 0 || depth >= max_depth) {
            throw runtime_error("SyntheticTreeEnv: cannot expand terminal state");
        }
        if (index < 0 || index >= get_num_nodes_at_depth(depth)) {
            throw runtime_error("SyntheticTreeEnv: invalid state index");
        }
        const int next_depth = depth + 1;
        const long long next_index_ll =
            static_cast<long long>(index) * static_cast<long long>(branching_factor)
            + static_cast<long long>(action);
        if (next_index_ll > numeric_limits<int>::max()) {
            throw runtime_error("SyntheticTreeEnv: child index overflow");
        }
        const int next_index = static_cast<int>(next_index_ll);
        if (next_index < 0 || next_index >= get_num_nodes_at_depth(next_depth)) {
            throw runtime_error("SyntheticTreeEnv: child index out of range");
        }
        return make_shared<mcts::IntPairState>(next_depth, next_index);
    }

    shared_ptr<mcts::IntPairStateDistr> SyntheticTreeEnv::get_transition_distribution(
        shared_ptr<const mcts::IntPairState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<mcts::IntPairStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }

        const double uniform_prob = action_noise / static_cast<double>(branching_factor);
        const double intended_bonus = 1.0 - action_noise;

        for (int a = 0; a < branching_factor; ++a) {
            double prob = uniform_prob;
            if (a == action->action) prob += intended_bonus;
            if (prob <= 0.0) continue;
            auto next_state = make_child_state(state, a);
            distr->insert_or_assign(next_state, prob);
        }

        double total_prob = 0.0;
        for (const auto& kv : *distr) total_prob += kv.second;
        if (total_prob > 0.0 && std::abs(total_prob - 1.0) > 1e-12) {
            for (auto& kv : *distr) kv.second /= total_prob;
        }

        return distr;
    }

    shared_ptr<const mcts::IntPairState> SyntheticTreeEnv::sample_transition_distribution(
        shared_ptr<const mcts::IntPairState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        if (is_sink_state(state)) return state;
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double SyntheticTreeEnv::reward_for_state(shared_ptr<const mcts::IntPairState> state) const {
        const int depth = state->state.first;
        const int index = state->state.second;
        if (depth != max_depth) return 0.0;
        const std::vector<double>* rewards = &leaf_rewards;
        if (resample_rewards_each_rollout && tls_ctx != nullptr && !tls_ctx->leaf_rewards.empty()) {
            rewards = &tls_ctx->leaf_rewards;
        }
        if (index < 0 || index >= static_cast<int>(rewards->size())) {
            throw runtime_error("SyntheticTreeEnv: leaf index out of range");
        }
        double val = (*rewards)[index];
        if (gamma != 1.0) {
            val *= pow(gamma, static_cast<double>(depth));
        }
        return val;
    }

    double SyntheticTreeEnv::get_reward(
        shared_ptr<const mcts::IntPairState> state,
        shared_ptr<const mcts::IntAction> action,
        shared_ptr<const mcts::IntPairState> observation) const
    {
        if (is_sink_state(state)) return 0.0;
        if (observation != nullptr) return reward_for_state(observation);

        double expected = 0.0;
        auto distr = get_transition_distribution(state, action);
        for (const auto& kv : *distr) {
            expected += kv.second * reward_for_state(kv.first);
        }
        return expected;
    }

    shared_ptr<const mcts::State> SyntheticTreeEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool SyntheticTreeEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        auto s = static_pointer_cast<const mcts::IntPairState>(state);
        return is_sink_state(s);
    }

    shared_ptr<mcts::ActionVector> SyntheticTreeEnv::get_valid_actions_itfc(
        shared_ptr<const mcts::State> state) const
    {
        auto s = static_pointer_cast<const mcts::IntPairState>(state);
        auto acts = get_valid_actions(s);
        auto out = make_shared<mcts::ActionVector>();
        for (const auto& a : *acts) out->push_back(static_pointer_cast<const mcts::Action>(a));
        return out;
    }

    shared_ptr<mcts::StateDistr> SyntheticTreeEnv::get_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action) const
    {
        auto s = static_pointer_cast<const mcts::IntPairState>(state);
        auto a = static_pointer_cast<const mcts::IntAction>(action);
        auto distr_typed = get_transition_distribution(s, a);
        auto out = make_shared<mcts::StateDistr>();
        for (auto& kv : *distr_typed) {
            out->insert_or_assign(static_pointer_cast<const mcts::State>(kv.first), kv.second);
        }
        return out;
    }

    shared_ptr<const mcts::State> SyntheticTreeEnv::sample_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        mcts::RandManager& rand_manager) const
    {
        auto s = static_pointer_cast<const mcts::IntPairState>(state);
        auto a = static_pointer_cast<const mcts::IntAction>(action);
        auto ns = sample_transition_distribution(s, a, rand_manager);
        return static_pointer_cast<const mcts::State>(ns);
    }

    double SyntheticTreeEnv::get_reward_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        shared_ptr<const mcts::Observation> observation) const
    {
        auto s = static_pointer_cast<const mcts::IntPairState>(state);
        auto a = static_pointer_cast<const mcts::IntAction>(action);
        auto o = static_pointer_cast<const mcts::IntPairState>(observation);
        return get_reward(s, a, o);
    }

    shared_ptr<mcts::MctsEnvContext> SyntheticTreeEnv::sample_context_itfc(
        shared_ptr<const mcts::State> state) const
    {
        (void)state;
        if (!resample_rewards_each_rollout) {
            tls_ctx = nullptr;
            return make_shared<mcts::MctsEnvContext>();
        }

        auto ctx = make_shared<SyntheticTreeEnvContext>();
        uint64_t rollout_id = rollout_counter.fetch_add(1, std::memory_order_relaxed);
        std::mt19937 rng(static_cast<uint32_t>(reward_seed + static_cast<int>(rollout_id)));
        build_leaf_rewards(rng, ctx->leaf_rewards);
        tls_ctx = ctx.get();
        return ctx;
    }
}
