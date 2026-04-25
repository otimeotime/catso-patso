#include "env/autonomous_vehicle_env.h"

#include "helper_templates.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

using namespace std;

namespace mcts::exp {
    AutonomousVehicleEnv::AutonomousVehicleEnv(
        vector<vector<int>> horizontal_edges,
        vector<vector<int>> vertical_edges,
        vector<array<double, num_reward_outcomes>> edge_rewards,
        array<double, 2> edge_probs,
        int max_steps,
        double reward_normalisation)
        : mcts::MctsEnv(true),
          horizontal_edges(move(horizontal_edges)),
          vertical_edges(move(vertical_edges)),
          edge_rewards(move(edge_rewards)),
          edge_probs(edge_probs),
          max_steps(max_steps),
          reward_normalisation(reward_normalisation),
          max_x(0),
          max_y(0)
    {
        if (this->horizontal_edges.empty() || this->horizontal_edges.front().empty()) {
            throw runtime_error("AutonomousVehicleEnv: horizontal_edges must be non-empty");
        }
        max_y = static_cast<int>(this->horizontal_edges.size());
        max_x = static_cast<int>(this->horizontal_edges.front().size()) + 1;

        if (max_x < 2 || max_y < 2) {
            throw runtime_error("AutonomousVehicleEnv: grid must be at least 2x2");
        }
        for (const auto& row : this->horizontal_edges) {
            if (static_cast<int>(row.size()) != max_x - 1) {
                throw runtime_error("AutonomousVehicleEnv: horizontal_edges has inconsistent row width");
            }
        }
        if (static_cast<int>(this->vertical_edges.size()) != max_y - 1) {
            throw runtime_error("AutonomousVehicleEnv: vertical_edges must have max_y - 1 rows");
        }
        for (const auto& row : this->vertical_edges) {
            if (static_cast<int>(row.size()) != max_x) {
                throw runtime_error("AutonomousVehicleEnv: vertical_edges has inconsistent row width");
            }
        }
        if (this->edge_rewards.empty()) {
            throw runtime_error("AutonomousVehicleEnv: edge_rewards must be non-empty");
        }
        if (this->edge_probs[0] < 0.0 || this->edge_probs[0] > this->edge_probs[1] || this->edge_probs[1] > 1.0) {
            throw runtime_error("AutonomousVehicleEnv: edge_probs must satisfy 0 <= p0 <= p1 <= 1");
        }
        if (this->max_steps <= 0) {
            throw runtime_error("AutonomousVehicleEnv: max_steps must be > 0");
        }
        if (this->reward_normalisation == 0.0) {
            throw runtime_error("AutonomousVehicleEnv: reward_normalisation must be non-zero");
        }

        const int num_edge_types = static_cast<int>(this->edge_rewards.size());
        auto validate_edge_type = [&](int edge_type) {
            if (edge_type < 0 || edge_type >= num_edge_types) {
                throw runtime_error("AutonomousVehicleEnv: edge type index out of range");
            }
        };

        for (const auto& row : this->horizontal_edges) {
            for (int edge_type : row) {
                validate_edge_type(edge_type);
            }
        }
        for (const auto& row : this->vertical_edges) {
            for (int edge_type : row) {
                validate_edge_type(edge_type);
            }
        }
    }

    vector<vector<int>> AutonomousVehicleEnv::default_horizontal_edges() {
        return {
            {1, 2, 0},
            {0, 2, 0},
            {0, 2, 1},
            {0, 0, 0}
        };
    }

    vector<vector<int>> AutonomousVehicleEnv::default_vertical_edges() {
        return {
            {0, 1, 0, 0},
            {0, 2, 2, 0},
            {0, 0, 1, 0}
        };
    }

    vector<array<double, AutonomousVehicleEnv::num_reward_outcomes>> AutonomousVehicleEnv::default_edge_rewards() {
        return {
            array<double, num_reward_outcomes>{6.0, 8.0, 10.0},
            array<double, num_reward_outcomes>{4.0, 12.0, 20.0},
            array<double, num_reward_outcomes>{2.0, 18.0, 40.0}
        };
    }

    array<double, 2> AutonomousVehicleEnv::default_edge_probs() {
        return {0.15, 0.85};
    }

    int AutonomousVehicleEnv::encode_time_outcome(int time, int outcome) const {
        return time * (num_reward_outcomes + 1) + outcome;
    }

    void AutonomousVehicleEnv::decode_state(
        shared_ptr<const mcts::Int3TupleState> state,
        int& x,
        int& y,
        int& time,
        int& last_outcome) const
    {
        x = get<0>(state->state);
        y = get<1>(state->state);
        const int encoded = get<2>(state->state);
        last_outcome = encoded % (num_reward_outcomes + 1);
        time = encoded / (num_reward_outcomes + 1);
    }

    shared_ptr<const mcts::Int3TupleState> AutonomousVehicleEnv::make_state(int x, int y, int time, int outcome) const {
        return make_shared<mcts::Int3TupleState>(x, y, encode_time_outcome(time, outcome));
    }

    bool AutonomousVehicleEnv::is_action_valid(int x, int y, int action) const {
        if (action == move_positive_y_action) {
            return y < max_y - 1;
        }
        if (action == move_positive_x_action) {
            return x < max_x - 1;
        }
        if (action == move_negative_y_action) {
            return y > 0;
        }
        if (action == move_negative_x_action) {
            return x > 0;
        }
        return false;
    }

    pair<int, int> AutonomousVehicleEnv::next_position(int x, int y, int action) const {
        if (!is_action_valid(x, y, action)) {
            throw runtime_error("AutonomousVehicleEnv: invalid action for current state");
        }

        if (action == move_positive_y_action) {
            return {x, y + 1};
        }
        if (action == move_positive_x_action) {
            return {x + 1, y};
        }
        if (action == move_negative_y_action) {
            return {x, y - 1};
        }
        return {x - 1, y};
    }

    int AutonomousVehicleEnv::get_edge_type(int x, int y, int action) const {
        if (!is_action_valid(x, y, action)) {
            throw runtime_error("AutonomousVehicleEnv: invalid action for edge lookup");
        }

        if (action == move_positive_y_action) {
            return vertical_edges[y][x];
        }
        if (action == move_positive_x_action) {
            return horizontal_edges[y][x];
        }
        if (action == move_negative_y_action) {
            return vertical_edges[y - 1][x];
        }
        return horizontal_edges[y][x - 1];
    }

    double AutonomousVehicleEnv::reward_for_transition(int edge_type, int outcome, bool reached_goal) const {
        if (edge_type < 0 || edge_type >= static_cast<int>(edge_rewards.size())) {
            throw runtime_error("AutonomousVehicleEnv: edge type out of range");
        }
        if (outcome < 0 || outcome >= num_reward_outcomes) {
            throw runtime_error("AutonomousVehicleEnv: outcome out of range");
        }

        double reward = -edge_rewards[edge_type][outcome];
        if (reached_goal) {
            reward += default_goal_bonus;
        }
        return reward / reward_normalisation;
    }

    shared_ptr<const mcts::Int3TupleState> AutonomousVehicleEnv::get_initial_state() const {
        return make_state(0, 0, 0, initial_outcome_tag);
    }

    bool AutonomousVehicleEnv::is_goal_state(shared_ptr<const mcts::Int3TupleState> state) const {
        int x = 0;
        int y = 0;
        int time = 0;
        int last_outcome = 0;
        decode_state(state, x, y, time, last_outcome);
        (void)time;
        (void)last_outcome;
        return x == max_x - 1 && y == max_y - 1;
    }

    bool AutonomousVehicleEnv::is_catastrophic_state(shared_ptr<const mcts::Int3TupleState> state) const {
        int x = 0;
        int y = 0;
        int time = 0;
        int last_outcome = 0;
        decode_state(state, x, y, time, last_outcome);
        (void)x;
        (void)y;
        (void)time;
        return last_outcome == (num_reward_outcomes - 1);
    }

    bool AutonomousVehicleEnv::is_sink_state(shared_ptr<const mcts::Int3TupleState> state) const {
        int x = 0;
        int y = 0;
        int time = 0;
        int last_outcome = 0;
        decode_state(state, x, y, time, last_outcome);
        (void)x;
        (void)y;
        (void)last_outcome;
        return time >= max_steps || is_goal_state(state);
    }

    shared_ptr<mcts::IntActionVector> AutonomousVehicleEnv::get_valid_actions(
        shared_ptr<const mcts::Int3TupleState> state) const
    {
        auto actions = make_shared<mcts::IntActionVector>();
        if (is_sink_state(state)) {
            return actions;
        }

        int x = 0;
        int y = 0;
        int time = 0;
        int last_outcome = 0;
        decode_state(state, x, y, time, last_outcome);
        (void)time;
        (void)last_outcome;

        for (int action = 0; action < num_actions; ++action) {
            if (is_action_valid(x, y, action)) {
                actions->push_back(make_shared<const mcts::IntAction>(action));
            }
        }
        return actions;
    }

    shared_ptr<mcts::Int3TupleStateDistr> AutonomousVehicleEnv::get_transition_distribution(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<mcts::Int3TupleStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }

        int x = 0;
        int y = 0;
        int time = 0;
        int last_outcome = 0;
        decode_state(state, x, y, time, last_outcome);
        (void)last_outcome;

        const auto [next_x, next_y] = next_position(x, y, action->action);
        const int next_time = time + 1;
        const double p0 = edge_probs[0];
        const double p1 = edge_probs[1] - edge_probs[0];
        const double p2 = 1.0 - edge_probs[1];

        if (p0 > 0.0) {
            distr->emplace(make_state(next_x, next_y, next_time, 0), p0);
        }
        if (p1 > 0.0) {
            distr->emplace(make_state(next_x, next_y, next_time, 1), p1);
        }
        if (p2 > 0.0) {
            distr->emplace(make_state(next_x, next_y, next_time, 2), p2);
        }
        return distr;
    }

    shared_ptr<const mcts::Int3TupleState> AutonomousVehicleEnv::sample_transition_distribution(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double AutonomousVehicleEnv::get_reward(
        shared_ptr<const mcts::Int3TupleState> state,
        shared_ptr<const mcts::IntAction> action,
        shared_ptr<const mcts::Int3TupleState> observation) const
    {
        if (is_sink_state(state)) {
            return 0.0;
        }

        int x = 0;
        int y = 0;
        int time = 0;
        int last_outcome = 0;
        decode_state(state, x, y, time, last_outcome);
        (void)time;
        (void)last_outcome;

        const int edge_type = get_edge_type(x, y, action->action);
        const auto [next_x, next_y] = next_position(x, y, action->action);
        const bool reached_goal = (next_x == max_x - 1) && (next_y == max_y - 1);

        if (observation != nullptr) {
            int observed_x = 0;
            int observed_y = 0;
            int observed_time = 0;
            int observed_outcome = 0;
            decode_state(observation, observed_x, observed_y, observed_time, observed_outcome);
            (void)observed_x;
            (void)observed_y;
            (void)observed_time;
            return reward_for_transition(edge_type, observed_outcome, reached_goal);
        }

        const double p0 = edge_probs[0];
        const double p1 = edge_probs[1] - edge_probs[0];
        const double p2 = 1.0 - edge_probs[1];
        return p0 * reward_for_transition(edge_type, 0, reached_goal)
            + p1 * reward_for_transition(edge_type, 1, reached_goal)
            + p2 * reward_for_transition(edge_type, 2, reached_goal);
    }

    shared_ptr<const mcts::State> AutonomousVehicleEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool AutonomousVehicleEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        auto typed_state = static_pointer_cast<const mcts::Int3TupleState>(state);
        return is_sink_state(typed_state);
    }

    bool AutonomousVehicleEnv::is_catastrophic_state_itfc(shared_ptr<const mcts::State> state) const {
        auto typed_state = static_pointer_cast<const mcts::Int3TupleState>(state);
        return is_catastrophic_state(typed_state);
    }

    shared_ptr<mcts::ActionVector> AutonomousVehicleEnv::get_valid_actions_itfc(
        shared_ptr<const mcts::State> state) const
    {
        auto typed_state = static_pointer_cast<const mcts::Int3TupleState>(state);
        auto typed_actions = get_valid_actions(typed_state);
        auto actions = make_shared<mcts::ActionVector>();
        for (const auto& typed_action : *typed_actions) {
            actions->push_back(static_pointer_cast<const mcts::Action>(typed_action));
        }
        return actions;
    }

    shared_ptr<mcts::StateDistr> AutonomousVehicleEnv::get_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action) const
    {
        auto typed_state = static_pointer_cast<const mcts::Int3TupleState>(state);
        auto typed_action = static_pointer_cast<const mcts::IntAction>(action);
        auto typed_distr = get_transition_distribution(typed_state, typed_action);
        auto distr = make_shared<mcts::StateDistr>();
        for (const auto& [next_state, probability] : *typed_distr) {
            distr->insert_or_assign(static_pointer_cast<const mcts::State>(next_state), probability);
        }
        return distr;
    }

    shared_ptr<const mcts::State> AutonomousVehicleEnv::sample_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        mcts::RandManager& rand_manager) const
    {
        auto typed_state = static_pointer_cast<const mcts::Int3TupleState>(state);
        auto typed_action = static_pointer_cast<const mcts::IntAction>(action);
        auto next_state = sample_transition_distribution(typed_state, typed_action, rand_manager);
        return static_pointer_cast<const mcts::State>(next_state);
    }

    double AutonomousVehicleEnv::get_reward_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        shared_ptr<const mcts::Observation> observation) const
    {
        auto typed_state = static_pointer_cast<const mcts::Int3TupleState>(state);
        auto typed_action = static_pointer_cast<const mcts::IntAction>(action);
        shared_ptr<const mcts::Int3TupleState> typed_observation = nullptr;
        if (observation != nullptr) {
            typed_observation = static_pointer_cast<const mcts::Int3TupleState>(observation);
        }
        return get_reward(typed_state, typed_action, typed_observation);
    }
}
