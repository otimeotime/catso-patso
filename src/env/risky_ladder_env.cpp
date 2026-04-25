#include "env/risky_ladder_env.h"

#include "helper_templates.h"

#include <memory>
#include <stdexcept>

using namespace std;

namespace mcts::exp {
    RiskyLadderEnv::RiskyLadderEnv(
        int H,
        double rung_reward,
        double bonus_reward,
        double fail_prob,
        double fail_penalty,
        int max_steps)
        : mcts::MctsEnv(true),
          H(H),
          rung_reward(rung_reward),
          bonus_reward(bonus_reward),
          fail_prob(fail_prob),
          fail_penalty(fail_penalty),
          max_steps(max_steps)
    {
        if (this->H < 2) {
            throw runtime_error("RiskyLadderEnv: H must be >= 2");
        }
        if (this->fail_prob < 0.0 || this->fail_prob > 1.0) {
            throw runtime_error("RiskyLadderEnv: fail_prob must be in [0,1]");
        }
        if (this->max_steps <= 0) {
            throw runtime_error("RiskyLadderEnv: max_steps must be > 0");
        }
    }

    void RiskyLadderEnv::decode_state(
        shared_ptr<const mcts::Int3TupleState> state,
        int& rung,
        int& steps,
        int& status) const
    {
        rung = get<0>(state->state);
        steps = get<1>(state->state);
        status = get<2>(state->state);
    }

    shared_ptr<const mcts::Int3TupleState> RiskyLadderEnv::make_state(int rung, int steps, int status) const {
        return make_shared<const mcts::Int3TupleState>(rung, steps, status);
    }

    shared_ptr<const mcts::Int3TupleState> RiskyLadderEnv::get_initial_state() const {
        return make_state(0, 0, active_status);
    }

    bool RiskyLadderEnv::is_sink_state(shared_ptr<const mcts::Int3TupleState> state) const {
        int rung = 0;
        int steps = 0;
        int status = 0;
        decode_state(state, rung, steps, status);
        (void)rung;
        (void)steps;
        return status != active_status;
    }

    shared_ptr<mcts::IntActionVector> RiskyLadderEnv::get_valid_actions(
        shared_ptr<const mcts::Int3TupleState> state) const
    {
        auto actions = make_shared<mcts::IntActionVector>();
        if (is_sink_state(state)) {
            return actions;
        }

        actions->push_back(make_shared<const mcts::IntAction>(advance_action));
        actions->push_back(make_shared<const mcts::IntAction>(bail_action));
        return actions;
    }

    shared_ptr<mcts::Int3TupleStateDistr> RiskyLadderEnv::get_transition_distribution(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<mcts::Int3TupleStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }
        if (action->action != advance_action && action->action != bail_action) {
            throw runtime_error("RiskyLadderEnv: action out of range");
        }

        int rung = 0;
        int steps = 0;
        int status = 0;
        decode_state(state, rung, steps, status);
        (void)status;
        const int next_steps = steps + 1;

        if (action->action == bail_action) {
            distr->insert_or_assign(make_state(rung, next_steps, bail_status), 1.0);
            return distr;
        }

        if (rung == H - 1) {
            distr->insert_or_assign(make_state(rung, next_steps, fail_status), fail_prob);
            distr->insert_or_assign(make_state(rung, next_steps, success_status), 1.0 - fail_prob);
            return distr;
        }

        const int next_status = (next_steps >= max_steps) ? truncated_status : active_status;
        distr->insert_or_assign(make_state(rung + 1, next_steps, next_status), 1.0);
        return distr;
    }

    shared_ptr<const mcts::Int3TupleState> RiskyLadderEnv::sample_transition_distribution(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double RiskyLadderEnv::get_reward(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        shared_ptr<const mcts::Int3TupleState> observation) const
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

        int rung = 0;
        int steps = 0;
        int status = 0;
        decode_state(state, rung, steps, status);
        (void)steps;
        (void)status;

        int next_rung = 0;
        int next_steps = 0;
        int next_status = 0;
        decode_state(observation, next_rung, next_steps, next_status);
        (void)next_rung;
        (void)next_steps;

        if (next_status == bail_status) {
            return rung_reward * static_cast<double>(rung);
        }
        if (next_status == success_status) {
            return rung_reward * static_cast<double>(rung) + bonus_reward;
        }
        if (next_status == fail_status) {
            return -fail_penalty;
        }
        return 0.0;
    }

    shared_ptr<const mcts::State> RiskyLadderEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool RiskyLadderEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_sink_state(static_pointer_cast<const mcts::Int3TupleState>(state));
    }

    shared_ptr<mcts::ActionVector> RiskyLadderEnv::get_valid_actions_itfc(
        shared_ptr<const mcts::State> state) const
    {
        auto typed_actions = get_valid_actions(static_pointer_cast<const mcts::Int3TupleState>(state));
        auto actions = make_shared<mcts::ActionVector>();
        for (const auto& typed_action : *typed_actions) {
            actions->push_back(static_pointer_cast<const mcts::Action>(typed_action));
        }
        return actions;
    }

    shared_ptr<mcts::StateDistr> RiskyLadderEnv::get_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action) const
    {
        auto typed_distr = get_transition_distribution(
            static_pointer_cast<const mcts::Int3TupleState>(state),
            static_pointer_cast<const mcts::IntAction>(action));
        auto distr = make_shared<mcts::StateDistr>();
        for (const auto& [next_state, probability] : *typed_distr) {
            distr->insert_or_assign(static_pointer_cast<const mcts::State>(next_state), probability);
        }
        return distr;
    }

    shared_ptr<const mcts::State> RiskyLadderEnv::sample_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        mcts::RandManager& rand_manager) const
    {
        return static_pointer_cast<const mcts::State>(
            sample_transition_distribution(
                static_pointer_cast<const mcts::Int3TupleState>(state),
                static_pointer_cast<const mcts::IntAction>(action),
                rand_manager));
    }

    double RiskyLadderEnv::get_reward_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        shared_ptr<const mcts::Observation> observation) const
    {
        shared_ptr<const mcts::Int3TupleState> typed_observation = nullptr;
        if (observation != nullptr) {
            typed_observation = static_pointer_cast<const mcts::Int3TupleState>(observation);
        }
        return get_reward(
            static_pointer_cast<const mcts::Int3TupleState>(state),
            static_pointer_cast<const mcts::IntAction>(action),
            typed_observation);
    }
}
