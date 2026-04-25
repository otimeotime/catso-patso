#include "env/lunar_lander_env.h"

#include "helper_templates.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>

using namespace std;

namespace {
    constexpr array<double, mcts::exp::LunarLanderEnv::num_bonus_outcomes> kLandingBonusValues{{
        -130.0, -30.0, 70.0, 170.0, 270.0
    }};
    constexpr array<double, mcts::exp::LunarLanderEnv::num_bonus_outcomes> kLandingBonusProbabilities{{
        0.06136, 0.24477, 0.38774, 0.24477, 0.06136
    }};
}

namespace mcts::exp {
    size_t LunarLanderState::hash() const {
        size_t cur_hash = 0;
        cur_hash = mcts::helper::hash_combine(cur_hash, x);
        cur_hash = mcts::helper::hash_combine(cur_hash, y);
        cur_hash = mcts::helper::hash_combine(cur_hash, vx);
        cur_hash = mcts::helper::hash_combine(cur_hash, vy);
        cur_hash = mcts::helper::hash_combine(cur_hash, angle);
        cur_hash = mcts::helper::hash_combine(cur_hash, step);
        cur_hash = mcts::helper::hash_combine(cur_hash, status);
        return mcts::helper::hash_combine(cur_hash, last_bonus_outcome);
    }

    bool LunarLanderState::equals(const LunarLanderState& other) const {
        return x == other.x
            && y == other.y
            && vx == other.vx
            && vy == other.vy
            && angle == other.angle
            && step == other.step
            && status == other.status
            && last_bonus_outcome == other.last_bonus_outcome;
    }

    bool LunarLanderState::equals_itfc(const mcts::Observation& other) const {
        try {
            const LunarLanderState& oth = dynamic_cast<const LunarLanderState&>(other);
            return equals(oth);
        }
        catch (const bad_cast&) {
            return false;
        }
    }

    string LunarLanderState::get_pretty_print_string() const {
        stringstream ss;
        ss << "(" << x
           << "," << y
           << "," << vx
           << "," << vy
           << "," << angle
           << "," << step
           << "," << status
           << "," << last_bonus_outcome
           << ")";
        return ss.str();
    }

    LunarLanderEnv::LunarLanderEnv(int max_steps, double reward_normalisation)
        : mcts::MctsEnv(true),
          max_steps(max_steps),
          reward_normalisation(reward_normalisation)
    {
        if (this->max_steps <= 0) {
            throw runtime_error("LunarLanderEnv: max_steps must be > 0");
        }
        if (this->reward_normalisation == 0.0) {
            throw runtime_error("LunarLanderEnv: reward_normalisation must be non-zero");
        }
    }

    shared_ptr<const LunarLanderState> LunarLanderEnv::make_state(
        int x,
        int y,
        int vx,
        int vy,
        int angle,
        int step,
        int status,
        int last_bonus_outcome) const
    {
        return make_shared<const LunarLanderState>(
            clamp(x, -4, 4),
            clamp(y, 0, 8),
            clamp(vx, -3, 3),
            clamp(vy, -3, 3),
            clamp(angle, -2, 2),
            step,
            status,
            last_bonus_outcome);
    }

    double LunarLanderEnv::landing_bonus_value(int outcome_index) const {
        if (outcome_index < 0 || outcome_index >= num_bonus_outcomes) {
            throw runtime_error("LunarLanderEnv: bonus outcome index out of range");
        }
        return kLandingBonusValues[static_cast<size_t>(outcome_index)];
    }

    double LunarLanderEnv::landing_bonus_probability(int outcome_index) const {
        if (outcome_index < 0 || outcome_index >= num_bonus_outcomes) {
            throw runtime_error("LunarLanderEnv: bonus probability index out of range");
        }
        return kLandingBonusProbabilities[static_cast<size_t>(outcome_index)];
    }

    shared_ptr<const LunarLanderState> LunarLanderEnv::get_initial_state() const {
        return make_state(0, 8, 0, -1, 0, 0, active_status, neutral_bonus_outcome);
    }

    bool LunarLanderEnv::is_sink_state(shared_ptr<const LunarLanderState> state) const {
        return state->status != active_status || state->step >= max_steps;
    }

    bool LunarLanderEnv::is_catastrophic_state(shared_ptr<const LunarLanderState> state) const {
        return state->status == crashed_status;
    }

    shared_ptr<mcts::IntActionVector> LunarLanderEnv::get_valid_actions(
        shared_ptr<const LunarLanderState> state) const
    {
        auto actions = make_shared<mcts::IntActionVector>();
        if (is_sink_state(state)) {
            return actions;
        }

        for (int action = 0; action < num_actions; ++action) {
            actions->push_back(make_shared<const mcts::IntAction>(action));
        }
        return actions;
    }

    shared_ptr<LunarLanderEnv::LunarLanderStateDistr> LunarLanderEnv::get_transition_distribution(
        shared_ptr<const LunarLanderState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<LunarLanderStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }
        if (action->action < 0 || action->action >= num_actions) {
            throw runtime_error("LunarLanderEnv: action out of range");
        }

        int thrust_x = 0;
        int thrust_y = 0;
        int angle_delta = 0;
        if (action->action == left_engine_action) {
            thrust_x += 1;
            thrust_y += 1;
            angle_delta += 1;
        }
        else if (action->action == main_engine_action) {
            thrust_y += 2;
            angle_delta += (state->angle > 0 ? -1 : (state->angle < 0 ? 1 : 0));
        }
        else if (action->action == right_engine_action) {
            thrust_x -= 1;
            thrust_y += 1;
            angle_delta -= 1;
        }

        const int next_vx = clamp(state->vx + thrust_x, -3, 3);
        const int next_vy = clamp(state->vy - 1 + thrust_y, -3, 3);
        const int next_angle = clamp(state->angle + angle_delta, -2, 2);
        const int next_x = clamp(state->x + next_vx, -4, 4);
        const int next_y = clamp(state->y + next_vy, 0, 8);
        const int next_step = state->step + 1;

        int next_status = active_status;
        if (next_y == 0) {
            if (abs(next_x) <= 1 && abs(next_vx) <= 1 && abs(next_vy) <= 1 && next_angle == 0) {
                next_status = landed_status;
            }
            else {
                next_status = crashed_status;
            }
        }
        else if (next_step >= max_steps) {
            next_status = truncated_status;
        }

        if (next_status == landed_status && next_x > 0) {
            for (int outcome = 0; outcome < num_bonus_outcomes; ++outcome) {
                distr->insert_or_assign(
                    make_state(next_x, next_y, next_vx, next_vy, next_angle, next_step, next_status, outcome),
                    landing_bonus_probability(outcome));
            }
        }
        else {
            distr->insert_or_assign(
                make_state(next_x, next_y, next_vx, next_vy, next_angle, next_step, next_status, neutral_bonus_outcome),
                1.0);
        }

        return distr;
    }

    shared_ptr<const LunarLanderState> LunarLanderEnv::sample_transition_distribution(
        shared_ptr<const LunarLanderState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double LunarLanderEnv::get_reward(
        shared_ptr<const LunarLanderState> state,
        shared_ptr<const mcts::IntAction> action,
        shared_ptr<const LunarLanderState> observation) const
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

        double reward = 0.0;
        reward -= 2.0 * abs(observation->x);
        reward -= 1.0 * abs(observation->y - 1);
        reward -= 3.0 * abs(observation->vx);
        reward -= 4.0 * abs(observation->vy);
        reward -= 5.0 * abs(observation->angle);
        if (action->action != no_op_action) {
            reward -= 1.0;
        }

        if (observation->status == landed_status) {
            reward += 100.0;
            if (observation->x > 0) {
                reward += landing_bonus_value(observation->last_bonus_outcome);
            }
        }
        else if (observation->status == crashed_status) {
            reward -= 100.0;
        }

        return reward / reward_normalisation;
    }

    shared_ptr<const mcts::State> LunarLanderEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool LunarLanderEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_sink_state(static_pointer_cast<const LunarLanderState>(state));
    }

    bool LunarLanderEnv::is_catastrophic_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_catastrophic_state(static_pointer_cast<const LunarLanderState>(state));
    }

    shared_ptr<mcts::ActionVector> LunarLanderEnv::get_valid_actions_itfc(
        shared_ptr<const mcts::State> state) const
    {
        auto typed_actions = get_valid_actions(static_pointer_cast<const LunarLanderState>(state));
        auto actions = make_shared<mcts::ActionVector>();
        for (const auto& typed_action : *typed_actions) {
            actions->push_back(static_pointer_cast<const mcts::Action>(typed_action));
        }
        return actions;
    }

    shared_ptr<mcts::StateDistr> LunarLanderEnv::get_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action) const
    {
        auto typed_distr = get_transition_distribution(
            static_pointer_cast<const LunarLanderState>(state),
            static_pointer_cast<const mcts::IntAction>(action));
        auto distr = make_shared<mcts::StateDistr>();
        for (const auto& [next_state, probability] : *typed_distr) {
            distr->insert_or_assign(static_pointer_cast<const mcts::State>(next_state), probability);
        }
        return distr;
    }

    shared_ptr<const mcts::State> LunarLanderEnv::sample_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        mcts::RandManager& rand_manager) const
    {
        return static_pointer_cast<const mcts::State>(
            sample_transition_distribution(
                static_pointer_cast<const LunarLanderState>(state),
                static_pointer_cast<const mcts::IntAction>(action),
                rand_manager));
    }

    double LunarLanderEnv::get_reward_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        shared_ptr<const mcts::Observation> observation) const
    {
        shared_ptr<const LunarLanderState> typed_observation = nullptr;
        if (observation != nullptr) {
            typed_observation = static_pointer_cast<const LunarLanderState>(observation);
        }
        return get_reward(
            static_pointer_cast<const LunarLanderState>(state),
            static_pointer_cast<const mcts::IntAction>(action),
            typed_observation);
    }
}
