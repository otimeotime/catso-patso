#include "env/driving_sim_env.h"

#include "helper_templates.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace std;

namespace mcts::exp {
    size_t DrivingSimState::hash() const {
        size_t cur_hash = 0;
        cur_hash = mcts::helper::hash_combine(cur_hash, headway);
        cur_hash = mcts::helper::hash_combine(cur_hash, lane_offset);
        cur_hash = mcts::helper::hash_combine(cur_hash, speed_diff);
        cur_hash = mcts::helper::hash_combine(cur_hash, step);
        cur_hash = mcts::helper::hash_combine(cur_hash, prev_acc_tag);
        return mcts::helper::hash_combine(cur_hash, status);
    }

    bool DrivingSimState::equals(const DrivingSimState& other) const {
        return headway == other.headway
            && lane_offset == other.lane_offset
            && speed_diff == other.speed_diff
            && step == other.step
            && prev_acc_tag == other.prev_acc_tag
            && status == other.status;
    }

    bool DrivingSimState::equals_itfc(const mcts::Observation& other) const {
        try {
            const DrivingSimState& oth = dynamic_cast<const DrivingSimState&>(other);
            return equals(oth);
        }
        catch (const bad_cast&) {
            return false;
        }
    }

    string DrivingSimState::get_pretty_print_string() const {
        stringstream ss;
        ss << "(" << headway
           << "," << lane_offset
           << "," << speed_diff
           << "," << step
           << "," << prev_acc_tag
           << "," << status
           << ")";
        return ss.str();
    }

    DrivingSimEnv::DrivingSimEnv(
        bool nine_actions,
        int max_steps,
        int max_headway,
        int max_lane_offset,
        int max_speed_diff,
        double p_oil,
        array<double, 5> leader_probs,
        double reward_normalisation)
        : mcts::MctsEnv(true),
          nine_actions(nine_actions),
          max_steps(max_steps),
          max_headway(max_headway),
          max_lane_offset(max_lane_offset),
          max_speed_diff(max_speed_diff),
          p_oil(p_oil),
          reward_normalisation(reward_normalisation),
          leader_probs(leader_probs),
          agent_actions_map{},
          num_actions(nine_actions ? 9 : 5)
    {
        if (this->max_steps <= 0 || this->max_headway <= 0 || this->max_lane_offset <= 0 || this->max_speed_diff <= 0) {
            throw runtime_error("DrivingSimEnv: all discretisation bounds must be positive");
        }
        if (this->p_oil < 0.0 || this->p_oil > 1.0) {
            throw runtime_error("DrivingSimEnv: p_oil must be in [0,1]");
        }
        if (this->reward_normalisation == 0.0) {
            throw runtime_error("DrivingSimEnv: reward_normalisation must be non-zero");
        }

        double total_prob = 0.0;
        for (double probability : this->leader_probs) {
            if (probability < 0.0) {
                throw runtime_error("DrivingSimEnv: leader probabilities must be >= 0");
            }
            total_prob += probability;
        }
        if (abs(total_prob - 1.0) > 1e-9) {
            throw runtime_error("DrivingSimEnv: leader probabilities must sum to 1");
        }

        if (this->nine_actions) {
            agent_actions_map = {{
                {-2, -1}, {-2, 0}, {-2, 1},
                {0, -1}, {0, 0}, {0, 1},
                {1, -1}, {1, 0}, {1, 1}
            }};
        }
        else {
            agent_actions_map = {{
                {-2, 0}, {0, -1}, {0, 0}, {0, 1}, {1, 0},
                {0, 0}, {0, 0}, {0, 0}, {0, 0}
            }};
        }
    }

    shared_ptr<const DrivingSimState> DrivingSimEnv::make_state(
        int headway,
        int lane_offset,
        int speed_diff,
        int step,
        int prev_acc_tag,
        int status) const
    {
        return make_shared<const DrivingSimState>(
            clamp_headway(headway),
            clamp_lane(lane_offset),
            clamp_speed_diff(speed_diff),
            step,
            prev_acc_tag,
            status);
    }

    int DrivingSimEnv::clamp_headway(int value) const {
        return clamp(value, 0, max_headway);
    }

    int DrivingSimEnv::clamp_lane(int value) const {
        return clamp(value, -max_lane_offset, max_lane_offset);
    }

    int DrivingSimEnv::clamp_speed_diff(int value) const {
        return clamp(value, -max_speed_diff, max_speed_diff);
    }

    int DrivingSimEnv::encode_acc_tag(int acceleration_delta) const {
        if (acceleration_delta < 0) {
            return 0;
        }
        if (acceleration_delta > 0) {
            return 2;
        }
        return 1;
    }

    shared_ptr<const DrivingSimState> DrivingSimEnv::get_initial_state() const {
        return make_state(6, 0, 0, 0, 1, active_status);
    }

    bool DrivingSimEnv::is_sink_state(shared_ptr<const DrivingSimState> state) const {
        return state->status != active_status || state->step >= max_steps;
    }

    bool DrivingSimEnv::is_catastrophic_state(shared_ptr<const DrivingSimState> state) const {
        return state->status == collision_status || state->status == escaped_status;
    }

    shared_ptr<mcts::IntActionVector> DrivingSimEnv::get_valid_actions(
        shared_ptr<const DrivingSimState> state) const
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

    shared_ptr<DrivingSimEnv::DrivingSimStateDistr> DrivingSimEnv::get_transition_distribution(
        shared_ptr<const DrivingSimState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<DrivingSimStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }
        if (action->action < 0 || action->action >= num_actions) {
            throw runtime_error("DrivingSimEnv: action out of range");
        }

        const auto [base_agent_acc, agent_lane_delta] = agent_actions_map[static_cast<size_t>(action->action)];
        vector<pair<int, double>> agent_acc_branches;
        if (base_agent_acc != 0 && p_oil > 0.0) {
            const int reduced_acc = (base_agent_acc < 0) ? -1 : 0;
            agent_acc_branches.push_back({base_agent_acc, 1.0 - p_oil});
            agent_acc_branches.push_back({reduced_acc, p_oil});
        }
        else {
            agent_acc_branches.push_back({base_agent_acc, 1.0});
        }

        vector<tuple<int, int, double>> leader_branches = {
            {0, 0, leader_probs[0]},
            {1, 0, leader_probs[1]},
            {-1, 0, leader_probs[2]},
            {-2, 0, leader_probs[3]},
            {0, -1, 0.5 * leader_probs[4]},
            {0, 1, 0.5 * leader_probs[4]}
        };

        for (const auto& [agent_acc, agent_prob] : agent_acc_branches) {
            const int next_prev_acc_tag = encode_acc_tag(agent_acc);
            for (const auto& [leader_acc, leader_lane_delta, leader_prob] : leader_branches) {
                if (leader_prob <= 0.0) {
                    continue;
                }

                const int next_speed_diff = clamp_speed_diff(state->speed_diff + leader_acc - agent_acc);
                const int next_headway = clamp_headway(state->headway + next_speed_diff);
                const int next_lane = clamp_lane(state->lane_offset + leader_lane_delta - agent_lane_delta);
                const int next_step = state->step + 1;

                int next_status = active_status;
                if (next_headway <= 0 && next_lane == 0) {
                    next_status = collision_status;
                }
                else if (abs(next_lane) >= max_lane_offset || next_headway >= max_headway) {
                    next_status = escaped_status;
                }
                else if (next_step >= max_steps) {
                    next_status = truncated_status;
                }

                const auto next_state = make_state(
                    next_headway,
                    next_lane,
                    next_speed_diff,
                    next_step,
                    next_prev_acc_tag,
                    next_status);
                (*distr)[next_state] += agent_prob * leader_prob;
            }
        }

        return distr;
    }

    shared_ptr<const DrivingSimState> DrivingSimEnv::sample_transition_distribution(
        shared_ptr<const DrivingSimState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double DrivingSimEnv::get_reward(
        shared_ptr<const DrivingSimState> state,
        shared_ptr<const mcts::IntAction> action,
        shared_ptr<const DrivingSimState> observation) const
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

        const auto [agent_acc, unused_lane_delta] = agent_actions_map[static_cast<size_t>(action->action)];
        (void)unused_lane_delta;
        const int desired_headway = 4;
        double reward = 0.0;
        reward -= 2.0 * abs(observation->headway - desired_headway);
        reward -= 3.0 * abs(observation->speed_diff);
        reward -= 4.0 * abs(observation->lane_offset);
        reward -= 1.5 * abs(encode_acc_tag(agent_acc) - state->prev_acc_tag);

        if (observation->status == collision_status) {
            reward -= 120.0;
        }
        else if (observation->status == escaped_status) {
            reward -= 40.0;
        }

        return reward / reward_normalisation;
    }

    shared_ptr<const mcts::State> DrivingSimEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool DrivingSimEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_sink_state(static_pointer_cast<const DrivingSimState>(state));
    }

    bool DrivingSimEnv::is_catastrophic_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_catastrophic_state(static_pointer_cast<const DrivingSimState>(state));
    }

    shared_ptr<mcts::ActionVector> DrivingSimEnv::get_valid_actions_itfc(
        shared_ptr<const mcts::State> state) const
    {
        auto typed_actions = get_valid_actions(static_pointer_cast<const DrivingSimState>(state));
        auto actions = make_shared<mcts::ActionVector>();
        for (const auto& typed_action : *typed_actions) {
            actions->push_back(static_pointer_cast<const mcts::Action>(typed_action));
        }
        return actions;
    }

    shared_ptr<mcts::StateDistr> DrivingSimEnv::get_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action) const
    {
        auto typed_distr = get_transition_distribution(
            static_pointer_cast<const DrivingSimState>(state),
            static_pointer_cast<const mcts::IntAction>(action));
        auto distr = make_shared<mcts::StateDistr>();
        for (const auto& [next_state, probability] : *typed_distr) {
            distr->insert_or_assign(static_pointer_cast<const mcts::State>(next_state), probability);
        }
        return distr;
    }

    shared_ptr<const mcts::State> DrivingSimEnv::sample_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        mcts::RandManager& rand_manager) const
    {
        return static_pointer_cast<const mcts::State>(
            sample_transition_distribution(
                static_pointer_cast<const DrivingSimState>(state),
                static_pointer_cast<const mcts::IntAction>(action),
                rand_manager));
    }

    double DrivingSimEnv::get_reward_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        shared_ptr<const mcts::Observation> observation) const
    {
        shared_ptr<const DrivingSimState> typed_observation = nullptr;
        if (observation != nullptr) {
            typed_observation = static_pointer_cast<const DrivingSimState>(observation);
        }
        return get_reward(
            static_pointer_cast<const DrivingSimState>(state),
            static_pointer_cast<const mcts::IntAction>(action),
            typed_observation);
    }
}
