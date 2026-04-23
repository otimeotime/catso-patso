#include "env/betting_game_env.h"

#include "helper_templates.h"

#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>

using namespace std;

namespace {
    constexpr double kBankrollEps = 1e-12;
}

namespace mcts::exp {
    size_t BettingGameState::hash() const {
        size_t cur_hash = 0;
        cur_hash = mcts::helper::hash_combine(cur_hash, bankroll);
        return mcts::helper::hash_combine(cur_hash, time);
    }

    bool BettingGameState::equals(const BettingGameState& other) const {
        return bankroll == other.bankroll && time == other.time;
    }

    bool BettingGameState::equals_itfc(const mcts::Observation& other) const {
        try {
            const BettingGameState& oth = dynamic_cast<const BettingGameState&>(other);
            return equals(oth);
        }
        catch (const bad_cast&) {
            return false;
        }
    }

    string BettingGameState::get_pretty_print_string() const {
        stringstream ss;
        ss << "(" << fixed << setprecision(6) << bankroll << "," << time << ")";
        return ss.str();
    }

    BettingGameEnv::BettingGameEnv(
        double win_prob,
        int max_sequence_length,
        double max_state_value,
        double initial_state,
        double reward_normalisation) :
            mcts::MctsEnv(true),
            win_prob(win_prob),
            max_sequence_length(max_sequence_length),
            max_state_value(max_state_value),
            initial_state(initial_state),
            reward_normalisation(reward_normalisation)
    {
        if (win_prob < 0.0 || win_prob > 1.0) {
            throw runtime_error("BettingGameEnv: win_prob must be in [0,1]");
        }
        if (max_sequence_length < 0) {
            throw runtime_error("BettingGameEnv: max_sequence_length must be >= 0");
        }
        if (max_state_value <= 0.0) {
            throw runtime_error("BettingGameEnv: max_state_value must be > 0");
        }
        if (initial_state < 0.0 || initial_state > max_state_value) {
            throw runtime_error("BettingGameEnv: initial_state must lie in [0, max_state_value]");
        }
        if (reward_normalisation == 0.0) {
            throw runtime_error("BettingGameEnv: reward_normalisation must be non-zero");
        }
    }

    shared_ptr<const BettingGameState> BettingGameEnv::make_state(double bankroll, int time) const {
        return make_shared<const BettingGameState>(bankroll, time);
    }

    double BettingGameEnv::compute_bet(
        shared_ptr<const BettingGameState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        if (action->action < 0 || action->action >= num_actions) {
            throw runtime_error("BettingGameEnv: action out of range");
        }

        const double action_fraction = static_cast<double>(action->action) / 8.0;
        return min(state->bankroll * action_fraction, max_state_value - state->bankroll);
    }

    bool BettingGameEnv::is_absorbing_bankroll(double bankroll) const {
        return bankroll <= kBankrollEps || bankroll >= max_state_value - kBankrollEps;
    }

    shared_ptr<const BettingGameState> BettingGameEnv::get_initial_state() const {
        return make_state(initial_state, 0);
    }

    bool BettingGameEnv::is_sink_state(shared_ptr<const BettingGameState> state) const {
        return state->time >= max_sequence_length || is_absorbing_bankroll(state->bankroll);
    }

    shared_ptr<mcts::IntActionVector> BettingGameEnv::get_valid_actions(
        shared_ptr<const BettingGameState> state) const
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

    shared_ptr<BettingGameEnv::BettingGameStateDistr> BettingGameEnv::get_transition_distribution(
        shared_ptr<const BettingGameState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<BettingGameStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }

        const double bet = compute_bet(state, action);
        const int next_time = state->time + 1;
        if (abs(bet) <= kBankrollEps) {
            distr->insert_or_assign(make_state(state->bankroll, next_time), 1.0);
            return distr;
        }

        const auto win_state = make_state(state->bankroll + bet, next_time);
        const auto lose_state = make_state(state->bankroll - bet, next_time);

        distr->insert_or_assign(win_state, win_prob);
        distr->insert_or_assign(lose_state, 1.0 - win_prob);
        return distr;
    }

    shared_ptr<const BettingGameState> BettingGameEnv::sample_transition_distribution(
        shared_ptr<const BettingGameState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        if (is_sink_state(state)) {
            return state;
        }

        const double bet = compute_bet(state, action);
        const int next_time = state->time + 1;
        const bool did_win = rand_manager.get_rand_uniform() < win_prob;
        const double next_bankroll = state->bankroll + (did_win ? bet : -bet);
        return make_state(next_bankroll, next_time);
    }

    double BettingGameEnv::get_reward(
        shared_ptr<const BettingGameState> state,
        shared_ptr<const mcts::IntAction> action,
        shared_ptr<const BettingGameState> observation) const
    {
        if (is_sink_state(state)) {
            return 0.0;
        }

        if (observation != nullptr) {
            return (observation->bankroll - state->bankroll) / reward_normalisation;
        }

        const double bet = compute_bet(state, action);
        return ((2.0 * win_prob) - 1.0) * bet / reward_normalisation;
    }

    shared_ptr<const mcts::State> BettingGameEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool BettingGameEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        auto s = static_pointer_cast<const BettingGameState>(state);
        return is_sink_state(s);
    }

    shared_ptr<mcts::ActionVector> BettingGameEnv::get_valid_actions_itfc(shared_ptr<const mcts::State> state) const {
        auto s = static_pointer_cast<const BettingGameState>(state);
        auto acts = get_valid_actions(s);
        auto out = make_shared<mcts::ActionVector>();
        for (const auto& a : *acts) {
            out->push_back(static_pointer_cast<const mcts::Action>(a));
        }
        return out;
    }

    shared_ptr<mcts::StateDistr> BettingGameEnv::get_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action) const
    {
        auto s = static_pointer_cast<const BettingGameState>(state);
        auto a = static_pointer_cast<const mcts::IntAction>(action);
        auto distr_typed = get_transition_distribution(s, a);
        auto out = make_shared<mcts::StateDistr>();
        for (const auto& kv : *distr_typed) {
            out->insert_or_assign(static_pointer_cast<const mcts::State>(kv.first), kv.second);
        }
        return out;
    }

    shared_ptr<const mcts::State> BettingGameEnv::sample_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        mcts::RandManager& rand_manager) const
    {
        auto s = static_pointer_cast<const BettingGameState>(state);
        auto a = static_pointer_cast<const mcts::IntAction>(action);
        auto ns = sample_transition_distribution(s, a, rand_manager);
        return static_pointer_cast<const mcts::State>(ns);
    }

    double BettingGameEnv::get_reward_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        shared_ptr<const mcts::Observation> observation) const
    {
        auto s = static_pointer_cast<const BettingGameState>(state);
        auto a = static_pointer_cast<const mcts::IntAction>(action);
        shared_ptr<const BettingGameState> o = nullptr;
        if (observation != nullptr) {
            o = static_pointer_cast<const BettingGameState>(observation);
        }
        return get_reward(s, a, o);
    }
}
