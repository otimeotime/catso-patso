#include "env/gambler_jackpot_env.h"

#include "env/discrete_env_utils.h"
#include "helper_templates.h"

#include <memory>
#include <stdexcept>

using namespace std;

namespace mcts::exp {
    GamblerJackpotEnv::GamblerJackpotEnv(
        int capital_target,
        double p_safe,
        double p_risky,
        int jackpot_gain,
        double jackpot_prob,
        double success_bonus,
        double bankruptcy_penalty,
        int max_steps)
        : mcts::MctsEnv(true),
          capital_target(capital_target),
          p_safe(p_safe),
          p_risky(p_risky),
          jackpot_gain(jackpot_gain),
          jackpot_prob(jackpot_prob),
          success_bonus(success_bonus),
          bankruptcy_penalty(bankruptcy_penalty),
          max_steps(max_steps)
    {
        if (this->capital_target < 2) {
            throw runtime_error("GamblerJackpotEnv: capital_target must be >= 2");
        }
        if (this->p_safe < 0.0 || this->p_safe > 1.0) {
            throw runtime_error("GamblerJackpotEnv: p_safe must be in [0,1]");
        }
        if (this->p_risky < 0.0 || this->p_risky > 1.0) {
            throw runtime_error("GamblerJackpotEnv: p_risky must be in [0,1]");
        }
        if (this->jackpot_prob < 0.0 || this->jackpot_prob > 1.0) {
            throw runtime_error("GamblerJackpotEnv: jackpot_prob must be in [0,1]");
        }
        if (this->max_steps <= 0) {
            throw runtime_error("GamblerJackpotEnv: max_steps must be > 0");
        }
    }

    void GamblerJackpotEnv::decode_state(
        shared_ptr<const mcts::Int3TupleState> state,
        int& capital,
        int& steps,
        int& status) const
    {
        capital = get<0>(state->state);
        steps = get<1>(state->state);
        status = get<2>(state->state);
    }

    shared_ptr<const mcts::Int3TupleState> GamblerJackpotEnv::make_state(int capital, int steps, int status) const {
        return make_shared<const mcts::Int3TupleState>(capital, steps, status);
    }

    shared_ptr<const mcts::Int3TupleState> GamblerJackpotEnv::get_initial_state() const {
        return make_state(capital_target / 2, 0, active_status);
    }

    bool GamblerJackpotEnv::is_sink_state(shared_ptr<const mcts::Int3TupleState> state) const {
        int capital = 0;
        int steps = 0;
        int status = 0;
        decode_state(state, capital, steps, status);
        (void)capital;
        (void)steps;
        return status != active_status;
    }

    shared_ptr<mcts::IntActionVector> GamblerJackpotEnv::get_valid_actions(
        shared_ptr<const mcts::Int3TupleState> state) const
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

    shared_ptr<mcts::Int3TupleStateDistr> GamblerJackpotEnv::get_transition_distribution(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<mcts::Int3TupleStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }
        if (action->action < 0 || action->action >= num_actions) {
            throw runtime_error("GamblerJackpotEnv: action out of range");
        }

        int capital = 0;
        int steps = 0;
        int status = 0;
        decode_state(state, capital, steps, status);
        (void)status;
        const int next_steps = steps + 1;

        auto make_outcome = [&](int next_capital, double probability) {
            int next_status = active_status;
            if (next_capital <= 0) {
                next_status = bankrupt_status;
            } else if (next_capital >= capital_target) {
                next_status = success_status;
            } else if (next_steps >= max_steps) {
                next_status = truncated_status;
            }
            detail::accumulate_probability(distr, make_state(next_capital, next_steps, next_status), probability);
        };

        if (action->action == 0) {
            make_outcome(capital + 1, p_safe);
            make_outcome(capital - 1, 1.0 - p_safe);
        } else if (action->action == 1) {
            make_outcome(capital + 2, p_risky);
            make_outcome(capital - 2, 1.0 - p_risky);
        } else {
            make_outcome(capital + jackpot_gain, jackpot_prob);
            make_outcome(capital - 2, 1.0 - jackpot_prob);
        }

        return distr;
    }

    shared_ptr<const mcts::Int3TupleState> GamblerJackpotEnv::sample_transition_distribution(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double GamblerJackpotEnv::get_reward(
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

        int capital = 0;
        int steps = 0;
        int status = 0;
        decode_state(state, capital, steps, status);
        (void)steps;
        (void)status;

        int next_capital = 0;
        int next_steps = 0;
        int next_status = 0;
        decode_state(observation, next_capital, next_steps, next_status);
        (void)next_steps;

        double reward = static_cast<double>(next_capital - capital);
        if (next_status == bankrupt_status) {
            reward += bankruptcy_penalty;
        } else if (next_status == success_status) {
            reward += success_bonus;
        }
        return reward;
    }

    shared_ptr<const mcts::State> GamblerJackpotEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool GamblerJackpotEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_sink_state(static_pointer_cast<const mcts::Int3TupleState>(state));
    }

    shared_ptr<mcts::ActionVector> GamblerJackpotEnv::get_valid_actions_itfc(
        shared_ptr<const mcts::State> state) const
    {
        auto typed_actions = get_valid_actions(static_pointer_cast<const mcts::Int3TupleState>(state));
        auto actions = make_shared<mcts::ActionVector>();
        for (const auto& typed_action : *typed_actions) {
            actions->push_back(static_pointer_cast<const mcts::Action>(typed_action));
        }
        return actions;
    }

    shared_ptr<mcts::StateDistr> GamblerJackpotEnv::get_transition_distribution_itfc(
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

    shared_ptr<const mcts::State> GamblerJackpotEnv::sample_transition_distribution_itfc(
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

    double GamblerJackpotEnv::get_reward_itfc(
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
