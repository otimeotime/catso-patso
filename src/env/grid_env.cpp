#include "env/grid_env.h"

#include <memory>

using namespace std;

namespace mcts::exp {
    shared_ptr<const mcts::IntPairState> GridEnv::get_initial_state() const {
        return make_shared<mcts::IntPairState>(0,0);
    }

    bool GridEnv::is_sink_state(shared_ptr<const mcts::IntPairState> state) const {
        return state->state == make_pair(grid_size, grid_size);
    }

    shared_ptr<mcts::StringActionVector> GridEnv::get_valid_actions(shared_ptr<const mcts::IntPairState> state) const {
        auto valid_actions = make_shared<mcts::StringActionVector>();
        if (is_sink_state(state)) return valid_actions;
        int x = state->state.first;
        int y = state->state.second;
        if (x > 0) valid_actions->push_back(make_shared<const mcts::StringAction>("left"));
        if (x < grid_size) valid_actions->push_back(make_shared<const mcts::StringAction>("right"));
        if (y > 0) valid_actions->push_back(make_shared<const mcts::StringAction>("up"));
        if (y < grid_size) valid_actions->push_back(make_shared<const mcts::StringAction>("down"));
        return valid_actions;
    }

    shared_ptr<const mcts::IntPairState> GridEnv::make_candidate_next_state(
        shared_ptr<const mcts::IntPairState> state,
        shared_ptr<const mcts::StringAction> action) const
    {
        auto new_state = make_shared<mcts::IntPairState>(state->state);
        if (action->action == "left") new_state->state.first -= 1;
        else if (action->action == "right") new_state->state.first += 1;
        else if (action->action == "up") new_state->state.second -= 1;
        else if (action->action == "down") new_state->state.second += 1;
        return new_state;
    }

    shared_ptr<mcts::IntPairStateDistr> GridEnv::get_transition_distribution(
        shared_ptr<const mcts::IntPairState> state,
        shared_ptr<const mcts::StringAction> action) const
    {
        auto next_state = make_candidate_next_state(state, action);
        auto distr = make_shared<mcts::IntPairStateDistr>();
        distr->insert_or_assign(next_state, 1.0 - stay_prob);
        if (stay_prob > 0.0) distr->insert_or_assign(state, stay_prob);
        return distr;
    }

    shared_ptr<const mcts::IntPairState> GridEnv::sample_transition_distribution(
        shared_ptr<const mcts::IntPairState> state,
        shared_ptr<const mcts::StringAction> action,
        mcts::RandManager& rand_manager) const
    {
        if (stay_prob > 0.0 && rand_manager.get_rand_uniform() < stay_prob) return state;
        return make_candidate_next_state(state, action);
    }

    double GridEnv::get_reward(
        shared_ptr<const mcts::IntPairState> state,
        shared_ptr<const mcts::StringAction> action,
        shared_ptr<const mcts::IntPairState> observation) const
    {
        return -1.0;
    }

    // Interface
    shared_ptr<const mcts::State> GridEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool GridEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        auto s = static_pointer_cast<const mcts::IntPairState>(state);
        return is_sink_state(s);
    }

    shared_ptr<mcts::ActionVector> GridEnv::get_valid_actions_itfc(shared_ptr<const mcts::State> state) const {
        auto s = static_pointer_cast<const mcts::IntPairState>(state);
        auto acts = get_valid_actions(s);
        auto out = make_shared<mcts::ActionVector>();
        for (auto a : *acts) out->push_back(static_pointer_cast<const mcts::Action>(a));
        return out;
    }

    shared_ptr<mcts::StateDistr> GridEnv::get_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action) const
    {
        auto s = static_pointer_cast<const mcts::IntPairState>(state);
        auto a = static_pointer_cast<const mcts::StringAction>(action);
        auto distr_itfc = get_transition_distribution(s, a);
        auto out = make_shared<mcts::StateDistr>();
        for (auto& kv : *distr_itfc) out->insert_or_assign(static_pointer_cast<const mcts::State>(kv.first), kv.second);
        return out;
    }

    shared_ptr<const mcts::State> GridEnv::sample_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        mcts::RandManager& rand_manager) const
    {
        auto s = static_pointer_cast<const mcts::IntPairState>(state);
        auto a = static_pointer_cast<const mcts::StringAction>(action);
        auto ns = sample_transition_distribution(s, a, rand_manager);
        return static_pointer_cast<const mcts::State>(ns);
    }

    double GridEnv::get_reward_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        shared_ptr<const mcts::Observation> observation) const
    {
        auto s = static_pointer_cast<const mcts::IntPairState>(state);
        auto a = static_pointer_cast<const mcts::StringAction>(action);
        auto o = static_pointer_cast<const mcts::IntPairState>(observation);
        return get_reward(s, a, o);
    }
}