#include "env/overflow_queue_env.h"

#include "env/discrete_env_utils.h"
#include "helper_templates.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <unordered_map>

using namespace std;

namespace mcts::exp {
    OverflowQueueEnv::OverflowQueueEnv(
        int K,
        int S_max,
        double burst_prob,
        int normal_high,
        int burst_low,
        int burst_high,
        double hold_cost,
        double service_cost,
        double overflow_penalty,
        bool terminal_on_overflow,
        int max_steps)
        : mcts::MctsEnv(true),
          K(K),
          S_max(S_max),
          burst_prob(burst_prob),
          normal_high(normal_high),
          burst_low(burst_low),
          burst_high(burst_high),
          hold_cost(hold_cost),
          service_cost(service_cost),
          overflow_penalty(overflow_penalty),
          terminal_on_overflow(terminal_on_overflow),
          max_steps(max_steps)
    {
        if (this->K < 1) {
            throw runtime_error("OverflowQueueEnv: K must be >= 1");
        }
        if (this->S_max < 0) {
            throw runtime_error("OverflowQueueEnv: S_max must be >= 0");
        }
        if (this->burst_prob < 0.0 || this->burst_prob > 1.0) {
            throw runtime_error("OverflowQueueEnv: burst_prob must be in [0,1]");
        }
        if (this->normal_high < 0) {
            throw runtime_error("OverflowQueueEnv: normal_high must be >= 0");
        }
        if (this->burst_low < 0 || this->burst_high < this->burst_low) {
            throw runtime_error("OverflowQueueEnv: burst range is invalid");
        }
        if (this->max_steps <= 0) {
            throw runtime_error("OverflowQueueEnv: max_steps must be > 0");
        }
    }

    void OverflowQueueEnv::decode_state(
        shared_ptr<const mcts::Int4TupleState> state,
        int& queue_len,
        int& steps,
        int& status,
        int& last_arrivals) const
    {
        queue_len = get<0>(state->state);
        steps = get<1>(state->state);
        status = get<2>(state->state);
        last_arrivals = get<3>(state->state);
    }

    shared_ptr<const mcts::Int4TupleState> OverflowQueueEnv::make_state(
        int queue_len,
        int steps,
        int status,
        int last_arrivals) const
    {
        return make_shared<const mcts::Int4TupleState>(queue_len, steps, status, last_arrivals);
    }

    shared_ptr<const mcts::Int4TupleState> OverflowQueueEnv::get_initial_state() const {
        return make_state(0, 0, active_status, 0);
    }

    bool OverflowQueueEnv::is_sink_state(shared_ptr<const mcts::Int4TupleState> state) const {
        int queue_len = 0;
        int steps = 0;
        int status = 0;
        int last_arrivals = 0;
        decode_state(state, queue_len, steps, status, last_arrivals);
        (void)queue_len;
        (void)steps;
        (void)last_arrivals;
        return status != active_status;
    }

    shared_ptr<mcts::IntActionVector> OverflowQueueEnv::get_valid_actions(
        shared_ptr<const mcts::Int4TupleState> state) const
    {
        auto actions = make_shared<mcts::IntActionVector>();
        if (is_sink_state(state)) {
            return actions;
        }

        for (int action = 0; action <= S_max; ++action) {
            actions->push_back(make_shared<const mcts::IntAction>(action));
        }
        return actions;
    }

    shared_ptr<mcts::Int4TupleStateDistr> OverflowQueueEnv::get_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<mcts::Int4TupleStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }
        if (action->action < 0 || action->action > S_max) {
            throw runtime_error("OverflowQueueEnv: action out of range");
        }

        int queue_len = 0;
        int steps = 0;
        int status = 0;
        int last_arrivals = 0;
        decode_state(state, queue_len, steps, status, last_arrivals);
        (void)status;
        (void)last_arrivals;
        const int service = min(action->action, queue_len);
        const int next_steps = steps + 1;

        unordered_map<int, double> arrival_probs;
        const double normal_prob = (1.0 - burst_prob) / static_cast<double>(normal_high + 1);
        for (int arrivals = 0; arrivals <= normal_high; ++arrivals) {
            arrival_probs[arrivals] += normal_prob;
        }
        const double burst_prob_per_value = burst_prob / static_cast<double>(burst_high - burst_low + 1);
        for (int arrivals = burst_low; arrivals <= burst_high; ++arrivals) {
            arrival_probs[arrivals] += burst_prob_per_value;
        }

        for (const auto& [arrivals, probability] : arrival_probs) {
            const int raw_next_queue = queue_len + arrivals - service;
            const int overflow_count = max(0, raw_next_queue - K);
            const int next_queue = min(K, max(0, raw_next_queue));

            int next_status = active_status;
            if (overflow_count > 0 && terminal_on_overflow) {
                next_status = overflow_terminal_status;
            } else if (next_steps >= max_steps) {
                next_status = truncated_status;
            }

            detail::accumulate_probability(
                distr,
                make_state(next_queue, next_steps, next_status, arrivals),
                probability);
        }

        return distr;
    }

    shared_ptr<const mcts::Int4TupleState> OverflowQueueEnv::sample_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double OverflowQueueEnv::get_reward(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        shared_ptr<const mcts::Int4TupleState> observation) const
    {
        if (is_sink_state(state)) {
            return 0.0;
        }

        if (observation == nullptr) {
            double expected_reward = 0.0;
            auto distr = get_transition_distribution(state, action);
            for (const auto& [next_state, probability] : *distr) {
                expected_reward += probability * get_reward(state, action, next_state);
            }
            return expected_reward;
        }

        int queue_len = 0;
        int steps = 0;
        int status = 0;
        int last_arrivals = 0;
        decode_state(state, queue_len, steps, status, last_arrivals);
        (void)steps;
        (void)status;
        (void)last_arrivals;

        int next_queue = 0;
        int next_steps = 0;
        int next_status = 0;
        int arrivals = 0;
        decode_state(observation, next_queue, next_steps, next_status, arrivals);
        (void)next_steps;
        (void)next_status;

        const int service = min(action->action, queue_len);
        const int overflow_count = max(0, queue_len + arrivals - service - K);
        double reward = hold_cost * static_cast<double>(next_queue)
            + service_cost * static_cast<double>(service);
        if (overflow_count > 0) {
            reward += overflow_penalty;
        }
        return reward;
    }

    shared_ptr<const mcts::State> OverflowQueueEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool OverflowQueueEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_sink_state(static_pointer_cast<const mcts::Int4TupleState>(state));
    }

    shared_ptr<mcts::ActionVector> OverflowQueueEnv::get_valid_actions_itfc(
        shared_ptr<const mcts::State> state) const
    {
        auto typed_actions = get_valid_actions(static_pointer_cast<const mcts::Int4TupleState>(state));
        auto actions = make_shared<mcts::ActionVector>();
        for (const auto& typed_action : *typed_actions) {
            actions->push_back(static_pointer_cast<const mcts::Action>(typed_action));
        }
        return actions;
    }

    shared_ptr<mcts::StateDistr> OverflowQueueEnv::get_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action) const
    {
        auto typed_distr = get_transition_distribution(
            static_pointer_cast<const mcts::Int4TupleState>(state),
            static_pointer_cast<const mcts::IntAction>(action));
        auto distr = make_shared<mcts::StateDistr>();
        for (const auto& [next_state, probability] : *typed_distr) {
            distr->insert_or_assign(static_pointer_cast<const mcts::State>(next_state), probability);
        }
        return distr;
    }

    shared_ptr<const mcts::State> OverflowQueueEnv::sample_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        mcts::RandManager& rand_manager) const
    {
        return static_pointer_cast<const mcts::State>(
            sample_transition_distribution(
                static_pointer_cast<const mcts::Int4TupleState>(state),
                static_pointer_cast<const mcts::IntAction>(action),
                rand_manager));
    }

    double OverflowQueueEnv::get_reward_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        shared_ptr<const mcts::Observation> observation) const
    {
        shared_ptr<const mcts::Int4TupleState> typed_observation = nullptr;
        if (observation != nullptr) {
            typed_observation = static_pointer_cast<const mcts::Int4TupleState>(observation);
        }
        return get_reward(
            static_pointer_cast<const mcts::Int4TupleState>(state),
            static_pointer_cast<const mcts::IntAction>(action),
            typed_observation);
    }
}
