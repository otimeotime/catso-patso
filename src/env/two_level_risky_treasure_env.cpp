#include "env/two_level_risky_treasure_env.h"

#include "helper_templates.h"

#include <memory>
#include <stdexcept>

using namespace std;

namespace mcts::exp {
    TwoLevelRiskyTreasureEnv::TwoLevelRiskyTreasureEnv(
        double safe_reward,
        double success_reward,
        double fail_penalty,
        double prior_success_prob,
        double hint_error,
        double abort_penalty)
        : mcts::MctsEnv(true),
          safe_reward(safe_reward),
          success_reward(success_reward),
          fail_penalty(fail_penalty),
          prior_success_prob(prior_success_prob),
          hint_error(hint_error),
          abort_penalty(abort_penalty)
    {
        if (this->prior_success_prob < 0.0 || this->prior_success_prob > 1.0) {
            throw runtime_error("TwoLevelRiskyTreasureEnv: prior_success_prob must be in [0,1]");
        }
        if (this->hint_error < 0.0 || this->hint_error > 1.0) {
            throw runtime_error("TwoLevelRiskyTreasureEnv: hint_error must be in [0,1]");
        }
    }

    void TwoLevelRiskyTreasureEnv::decode_state(
        shared_ptr<const mcts::Int4TupleState> state,
        int& stage,
        int& hint,
        int& steps,
        int& status) const
    {
        stage = get<0>(state->state);
        hint = get<1>(state->state);
        steps = get<2>(state->state);
        status = get<3>(state->state);
    }

    shared_ptr<const mcts::Int4TupleState> TwoLevelRiskyTreasureEnv::make_state(
        int stage,
        int hint,
        int steps,
        int status) const
    {
        return make_shared<const mcts::Int4TupleState>(stage, hint, steps, status);
    }

    double TwoLevelRiskyTreasureEnv::posterior_success_prob(int hint) const {
        if (hint == good_hint) {
            const double denom = prior_success_prob * (1.0 - hint_error)
                + (1.0 - prior_success_prob) * hint_error;
            return denom > 0.0 ? (prior_success_prob * (1.0 - hint_error)) / denom : prior_success_prob;
        }

        const double denom = prior_success_prob * hint_error
            + (1.0 - prior_success_prob) * (1.0 - hint_error);
        return denom > 0.0 ? (prior_success_prob * hint_error) / denom : prior_success_prob;
    }

    shared_ptr<const mcts::Int4TupleState> TwoLevelRiskyTreasureEnv::get_initial_state() const {
        return make_state(root_stage, no_hint, 0, active_status);
    }

    bool TwoLevelRiskyTreasureEnv::is_sink_state(shared_ptr<const mcts::Int4TupleState> state) const {
        int stage = 0;
        int hint = 0;
        int steps = 0;
        int status = 0;
        decode_state(state, stage, hint, steps, status);
        (void)stage;
        (void)hint;
        (void)steps;
        return status != active_status;
    }

    shared_ptr<mcts::IntActionVector> TwoLevelRiskyTreasureEnv::get_valid_actions(
        shared_ptr<const mcts::Int4TupleState> state) const
    {
        auto actions = make_shared<mcts::IntActionVector>();
        if (is_sink_state(state)) {
            return actions;
        }

        int stage = 0;
        int hint = 0;
        int steps = 0;
        int status = 0;
        decode_state(state, stage, hint, steps, status);
        (void)hint;
        (void)steps;
        (void)status;

        if (stage == root_stage) {
            actions->push_back(make_shared<const mcts::IntAction>(safe_action));
            actions->push_back(make_shared<const mcts::IntAction>(risky_action));
        } else {
            actions->push_back(make_shared<const mcts::IntAction>(abort_action));
            actions->push_back(make_shared<const mcts::IntAction>(proceed_action));
        }
        return actions;
    }

    shared_ptr<mcts::Int4TupleStateDistr> TwoLevelRiskyTreasureEnv::get_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<mcts::Int4TupleStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }

        int stage = 0;
        int hint = 0;
        int steps = 0;
        int status = 0;
        decode_state(state, stage, hint, steps, status);
        (void)status;
        const int next_steps = steps + 1;

        if (stage == root_stage) {
            if (action->action == safe_action) {
                distr->insert_or_assign(make_state(stage, no_hint, next_steps, safe_status), 1.0);
                return distr;
            }
            if (action->action != risky_action) {
                throw runtime_error("TwoLevelRiskyTreasureEnv: invalid root action");
            }

            const double p_good = prior_success_prob * (1.0 - hint_error)
                + (1.0 - prior_success_prob) * hint_error;
            distr->insert_or_assign(make_state(hint_stage, good_hint, next_steps, active_status), p_good);
            distr->insert_or_assign(make_state(hint_stage, bad_hint, next_steps, active_status), 1.0 - p_good);
            return distr;
        }

        if (action->action == abort_action) {
            distr->insert_or_assign(make_state(stage, hint, next_steps, abort_status), 1.0);
            return distr;
        }
        if (action->action != proceed_action) {
            throw runtime_error("TwoLevelRiskyTreasureEnv: invalid hint-stage action");
        }

        const double p_success = posterior_success_prob(hint);
        distr->insert_or_assign(make_state(stage, hint, next_steps, success_status), p_success);
        distr->insert_or_assign(make_state(stage, hint, next_steps, fail_status), 1.0 - p_success);
        return distr;
    }

    shared_ptr<const mcts::Int4TupleState> TwoLevelRiskyTreasureEnv::sample_transition_distribution(
        shared_ptr<const mcts::Int4TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double TwoLevelRiskyTreasureEnv::get_reward(
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

        int stage = 0;
        int hint = 0;
        int steps = 0;
        int status = 0;
        decode_state(observation, stage, hint, steps, status);
        (void)stage;
        (void)hint;
        (void)steps;

        if (status == safe_status) {
            return safe_reward;
        }
        if (status == abort_status) {
            return -abort_penalty;
        }
        if (status == success_status) {
            return success_reward;
        }
        if (status == fail_status) {
            return -fail_penalty;
        }
        return 0.0;
    }

    shared_ptr<const mcts::State> TwoLevelRiskyTreasureEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool TwoLevelRiskyTreasureEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_sink_state(static_pointer_cast<const mcts::Int4TupleState>(state));
    }

    shared_ptr<mcts::ActionVector> TwoLevelRiskyTreasureEnv::get_valid_actions_itfc(
        shared_ptr<const mcts::State> state) const
    {
        auto typed_actions = get_valid_actions(static_pointer_cast<const mcts::Int4TupleState>(state));
        auto actions = make_shared<mcts::ActionVector>();
        for (const auto& typed_action : *typed_actions) {
            actions->push_back(static_pointer_cast<const mcts::Action>(typed_action));
        }
        return actions;
    }

    shared_ptr<mcts::StateDistr> TwoLevelRiskyTreasureEnv::get_transition_distribution_itfc(
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

    shared_ptr<const mcts::State> TwoLevelRiskyTreasureEnv::sample_transition_distribution_itfc(
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

    double TwoLevelRiskyTreasureEnv::get_reward_itfc(
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
