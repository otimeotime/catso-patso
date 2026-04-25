#include "env/river_swim_stochastic_env.h"

#include "helper_templates.h"

#include <memory>
#include <stdexcept>

using namespace std;

namespace mcts::exp {
    RiverSwimStochasticEnv::RiverSwimStochasticEnv(
        int N,
        double p_right_success,
        double p_right_fail_left,
        double left_reward,
        double right_reward,
        int max_steps)
        : mcts::MctsEnv(true),
          N(N),
          p_right_success(p_right_success),
          p_right_fail_left(p_right_fail_left),
          left_reward(left_reward),
          right_reward(right_reward),
          max_steps(max_steps)
    {
        if (this->N < 3) {
            throw runtime_error("RiverSwimStochasticEnv: N must be >= 3");
        }
        if (this->p_right_success < 0.0 || this->p_right_success > 1.0) {
            throw runtime_error("RiverSwimStochasticEnv: p_right_success must be in [0,1]");
        }
        if (this->p_right_fail_left < 0.0 || this->p_right_fail_left > 1.0) {
            throw runtime_error("RiverSwimStochasticEnv: p_right_fail_left must be in [0,1]");
        }
        if (this->p_right_success + this->p_right_fail_left > 1.0) {
            throw runtime_error("RiverSwimStochasticEnv: right action probabilities must sum to <= 1");
        }
        if (this->max_steps <= 0) {
            throw runtime_error("RiverSwimStochasticEnv: max_steps must be > 0");
        }
    }

    void RiverSwimStochasticEnv::decode_state(
        shared_ptr<const mcts::Int4TupleState> state,
        int& river_state,
        int& steps,
        int& status,
        int& last_outcome) const
    {
        river_state = get<0>(state->state);
        steps = get<1>(state->state);
        status = get<2>(state->state);
        last_outcome = get<3>(state->state);
    }

    shared_ptr<const mcts::Int4TupleState> RiverSwimStochasticEnv::make_state(
        int river_state,
        int steps,
        int status,
        int last_outcome) const
    {
        return make_shared<const mcts::Int4TupleState>(river_state, steps, status, last_outcome);
    }

    shared_ptr<const mcts::Int4TupleState> RiverSwimStochasticEnv::get_initial_state() const {
        return make_state(0, 0, active_status, left_outcome);
    }

    bool RiverSwimStochasticEnv::is_sink_state(shared_ptr<const mcts::Int4TupleState> state) const {
        int river_state = 0;
        int steps = 0;
        int status = 0;
        int last_outcome = 0;
        decode_state(state, river_state, steps, status, last_outcome);
        (void)river_state;
        (void)steps;
        (void)last_outcome;
        return status != active_status;
    }

    shared_ptr<mcts::IntActionVector> RiverSwimStochasticEnv::get_valid_actions(
        shared_ptr<const mcts::Int4TupleState> state) const
    {
        auto actions = make_shared<mcts::IntActionVector>();
        if (is_sink_state(state)) {
            return actions;
        }

        actions->push_back(make_shared<const mcts::IntAction>(left_action));
        actions->push_back(make_shared<const mcts::IntAction>(right_action));
        return actions;
    }

    shared_ptr<mcts::Int4TupleStateDistr> RiverSwimStochasticEnv::get_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<mcts::Int4TupleStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }
        if (action->action != left_action && action->action != right_action) {
            throw runtime_error("RiverSwimStochasticEnv: action out of range");
        }

        int river_state = 0;
        int steps = 0;
        int status = 0;
        int last_outcome = 0;
        decode_state(state, river_state, steps, status, last_outcome);
        (void)status;
        (void)last_outcome;
        const int next_steps = steps + 1;
        const int next_status = (next_steps >= max_steps) ? truncated_status : active_status;

        if (action->action == left_action) {
            distr->insert_or_assign(make_state(max(0, river_state - 1), next_steps, next_status, left_outcome), 1.0);
            return distr;
        }

        const int success_state = min(N - 1, river_state + 1);
        const int fail_left_state = max(0, river_state - 1);
        distr->insert_or_assign(make_state(success_state, next_steps, next_status, right_success_outcome), p_right_success);
        distr->insert_or_assign(make_state(fail_left_state, next_steps, next_status, right_fail_left_outcome), p_right_fail_left);
        distr->insert_or_assign(
            make_state(river_state, next_steps, next_status, right_fail_stay_outcome),
            1.0 - p_right_success - p_right_fail_left);
        return distr;
    }

    shared_ptr<const mcts::Int4TupleState> RiverSwimStochasticEnv::sample_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double RiverSwimStochasticEnv::get_reward(
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

        int next_river_state = 0;
        int next_steps = 0;
        int next_status = 0;
        int last_outcome = 0;
        decode_state(observation, next_river_state, next_steps, next_status, last_outcome);
        (void)next_steps;
        (void)next_status;

        if (action->action == left_action && last_outcome == left_outcome && next_river_state == 0) {
            return left_reward;
        }
        if (action->action == right_action && last_outcome == right_success_outcome && next_river_state == N - 1) {
            return right_reward;
        }
        return 0.0;
    }

    shared_ptr<const mcts::State> RiverSwimStochasticEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool RiverSwimStochasticEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_sink_state(static_pointer_cast<const mcts::Int4TupleState>(state));
    }

    shared_ptr<mcts::ActionVector> RiverSwimStochasticEnv::get_valid_actions_itfc(
        shared_ptr<const mcts::State> state) const
    {
        auto typed_actions = get_valid_actions(static_pointer_cast<const mcts::Int4TupleState>(state));
        auto actions = make_shared<mcts::ActionVector>();
        for (const auto& typed_action : *typed_actions) {
            actions->push_back(static_pointer_cast<const mcts::Action>(typed_action));
        }
        return actions;
    }

    shared_ptr<mcts::StateDistr> RiverSwimStochasticEnv::get_transition_distribution_itfc(
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

    shared_ptr<const mcts::State> RiverSwimStochasticEnv::sample_transition_distribution_itfc(
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

    double RiverSwimStochasticEnv::get_reward_itfc(
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
