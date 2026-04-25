#include "env/servers_sim_env.h"

#include "helper_templates.h"

#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>

using namespace std;

namespace {
    constexpr double kPoissonMassFloor = 1e-12;
}

namespace mcts::exp {
    size_t ServersSimState::hash() const {
        size_t cur_hash = 0;
        cur_hash = mcts::helper::hash_combine(cur_hash, paid_servers);
        cur_hash = mcts::helper::hash_combine(cur_hash, booting_one);
        cur_hash = mcts::helper::hash_combine(cur_hash, booting_two);
        cur_hash = mcts::helper::hash_combine(cur_hash, queue_units);
        cur_hash = mcts::helper::hash_combine(cur_hash, peak_steps_remaining);
        return mcts::helper::hash_combine(cur_hash, step);
    }

    bool ServersSimState::equals(const ServersSimState& other) const {
        return paid_servers == other.paid_servers
            && booting_one == other.booting_one
            && booting_two == other.booting_two
            && queue_units == other.queue_units
            && peak_steps_remaining == other.peak_steps_remaining
            && step == other.step;
    }

    bool ServersSimState::equals_itfc(const mcts::Observation& other) const {
        try {
            const ServersSimState& oth = dynamic_cast<const ServersSimState&>(other);
            return equals(oth);
        }
        catch (const bad_cast&) {
            return false;
        }
    }

    string ServersSimState::get_pretty_print_string() const {
        stringstream ss;
        ss << "(" << paid_servers
           << "," << booting_one
           << "," << booting_two
           << "," << queue_units
           << "," << peak_steps_remaining
           << "," << step
           << ")";
        return ss.str();
    }

    ServersSimEnv::ServersSimEnv(
        int max_servers,
        int min_servers,
        int init_servers,
        int decision_horizon,
        int peak_duration_steps,
        int queue_unit_size,
        int service_units_per_server,
        int max_queue_units,
        int max_arrival_units,
        double normal_arrival_units_mean,
        double peak_arrival_units_mean,
        double peak_start_prob,
        double tts_cost,
        double server_cost,
        double reward_normalisation)
        : mcts::MctsEnv(true),
          max_servers(max_servers),
          min_servers(min_servers),
          init_servers(init_servers),
          decision_horizon(decision_horizon),
          peak_duration_steps(peak_duration_steps),
          queue_unit_size(queue_unit_size),
          service_units_per_server(service_units_per_server),
          max_queue_units(max_queue_units),
          max_arrival_units(max_arrival_units),
          catastrophic_queue_units(max(1, max_queue_units - 5)),
          normal_arrival_units_mean(normal_arrival_units_mean),
          peak_arrival_units_mean(peak_arrival_units_mean),
          peak_start_prob(peak_start_prob),
          tts_cost(tts_cost),
          server_cost(server_cost),
          reward_normalisation(reward_normalisation)
    {
        if (this->max_servers <= 0) {
            throw runtime_error("ServersSimEnv: max_servers must be > 0");
        }
        if (this->min_servers < 0 || this->min_servers > this->max_servers) {
            throw runtime_error("ServersSimEnv: min_servers must lie in [0, max_servers]");
        }
        if (this->init_servers < this->min_servers || this->init_servers > this->max_servers) {
            throw runtime_error("ServersSimEnv: init_servers must lie in [min_servers, max_servers]");
        }
        if (this->decision_horizon <= 0) {
            throw runtime_error("ServersSimEnv: decision_horizon must be > 0");
        }
        if (this->peak_duration_steps <= 0) {
            throw runtime_error("ServersSimEnv: peak_duration_steps must be > 0");
        }
        if (this->queue_unit_size <= 0 || this->service_units_per_server < 0) {
            throw runtime_error("ServersSimEnv: queue and service scaling must be positive");
        }
        if (this->max_queue_units <= 0 || this->max_arrival_units <= 0) {
            throw runtime_error("ServersSimEnv: max queue/arrival units must be positive");
        }
        if (this->normal_arrival_units_mean < 0.0 || this->peak_arrival_units_mean < 0.0) {
            throw runtime_error("ServersSimEnv: arrival means must be >= 0");
        }
        if (this->peak_start_prob < 0.0 || this->peak_start_prob > 1.0) {
            throw runtime_error("ServersSimEnv: peak_start_prob must be in [0,1]");
        }
        if (this->reward_normalisation == 0.0) {
            throw runtime_error("ServersSimEnv: reward_normalisation must be non-zero");
        }

        normal_arrival_distribution = build_truncated_poisson(this->normal_arrival_units_mean);
        peak_arrival_distribution = build_truncated_poisson(this->peak_arrival_units_mean);
    }

    shared_ptr<const ServersSimState> ServersSimEnv::make_state(
        int paid_servers,
        int booting_one,
        int booting_two,
        int queue_units,
        int peak_steps_remaining,
        int step) const
    {
        validate_state_components(paid_servers, booting_one, booting_two, queue_units, peak_steps_remaining, step);
        return make_shared<const ServersSimState>(
            paid_servers,
            booting_one,
            booting_two,
            queue_units,
            peak_steps_remaining,
            step);
    }

    void ServersSimEnv::validate_state_components(
        int paid_servers,
        int booting_one,
        int booting_two,
        int queue_units,
        int peak_steps_remaining,
        int step) const
    {
        if (paid_servers < min_servers || paid_servers > max_servers) {
            throw runtime_error("ServersSimEnv: paid_servers out of range");
        }
        if (booting_one < 0 || booting_two < 0 || booting_one + booting_two > paid_servers) {
            throw runtime_error("ServersSimEnv: booting counts out of range");
        }
        if (queue_units < 0 || queue_units > max_queue_units) {
            throw runtime_error("ServersSimEnv: queue_units out of range");
        }
        if (peak_steps_remaining < 0 || peak_steps_remaining > peak_duration_steps) {
            throw runtime_error("ServersSimEnv: peak_steps_remaining out of range");
        }
        if (step < 0 || step > decision_horizon) {
            throw runtime_error("ServersSimEnv: step out of range");
        }
    }

    vector<pair<int, double>> ServersSimEnv::build_truncated_poisson(double mean_units) const {
        vector<pair<int, double>> distribution;
        if (mean_units <= kPoissonMassFloor) {
            distribution.push_back({0, 1.0});
            return distribution;
        }

        double prob = std::exp(-mean_units);
        double total = 0.0;
        for (int arrivals = 0; arrivals < max_arrival_units; ++arrivals) {
            distribution.push_back({arrivals, prob});
            total += prob;
            prob *= mean_units / static_cast<double>(arrivals + 1);
        }
        distribution.push_back({max_arrival_units, max(0.0, 1.0 - total)});
        return distribution;
    }

    int ServersSimEnv::active_servers(shared_ptr<const ServersSimState> state) const {
        return state->paid_servers - state->booting_one - state->booting_two;
    }

    double ServersSimEnv::step_cost(shared_ptr<const ServersSimState> observation) const {
        const double server_step_cost = server_cost * static_cast<double>(observation->paid_servers)
            / static_cast<double>(decision_horizon);
        const double backlog_step_cost = tts_cost
            * static_cast<double>(observation->queue_units * queue_unit_size)
            / 60.0;
        return server_step_cost + backlog_step_cost;
    }

    shared_ptr<const ServersSimState> ServersSimEnv::get_initial_state() const {
        return make_state(init_servers, 0, 0, 0, 0, 0);
    }

    bool ServersSimEnv::is_sink_state(shared_ptr<const ServersSimState> state) const {
        return state->step >= decision_horizon;
    }

    bool ServersSimEnv::is_catastrophic_state(shared_ptr<const ServersSimState> state) const {
        return state->queue_units >= catastrophic_queue_units;
    }

    shared_ptr<mcts::IntActionVector> ServersSimEnv::get_valid_actions(
        shared_ptr<const ServersSimState> state) const
    {
        auto actions = make_shared<mcts::IntActionVector>();
        if (is_sink_state(state)) {
            return actions;
        }

        actions->push_back(make_shared<const mcts::IntAction>(keep_action));
        if (state->paid_servers < max_servers) {
            actions->push_back(make_shared<const mcts::IntAction>(add_server_action));
        }
        if (state->paid_servers > min_servers) {
            actions->push_back(make_shared<const mcts::IntAction>(remove_server_action));
        }
        return actions;
    }

    shared_ptr<ServersSimEnv::ServersSimStateDistr> ServersSimEnv::get_transition_distribution(
        shared_ptr<const ServersSimState> state,
        shared_ptr<const mcts::IntAction> action) const
    {
        auto distr = make_shared<ServersSimStateDistr>();
        if (is_sink_state(state)) {
            distr->insert_or_assign(state, 1.0);
            return distr;
        }
        if (action->action < 0 || action->action >= num_actions) {
            throw runtime_error("ServersSimEnv: action out of range");
        }

        int paid_servers = state->paid_servers;
        int booting_one = state->booting_one;
        int booting_two = state->booting_two;

        if (action->action == add_server_action && paid_servers < max_servers) {
            paid_servers++;
            booting_two++;
        }
        else if (action->action == remove_server_action && paid_servers > min_servers) {
            if (booting_two > 0) {
                booting_two--;
                paid_servers--;
            }
            else if (booting_one > 0) {
                booting_one--;
                paid_servers--;
            }
            else {
                paid_servers--;
            }
        }

        const int available_servers = paid_servers - booting_one - booting_two;
        const int service_units = max(0, available_servers) * service_units_per_server;
        const int next_booting_one = booting_two;
        const int next_booting_two = 0;
        const int next_step = state->step + 1;

        auto accumulate_regime = [&](const vector<pair<int, double>>& arrival_distribution, int next_peak_steps, double regime_prob) {
            for (const auto& [arrival_units, arrival_prob] : arrival_distribution) {
                const int next_queue = clamp(state->queue_units + arrival_units - service_units, 0, max_queue_units);
                const auto next_state = make_state(
                    paid_servers,
                    next_booting_one,
                    next_booting_two,
                    next_queue,
                    next_peak_steps,
                    next_step);
                (*distr)[next_state] += regime_prob * arrival_prob;
            }
        };

        if (state->peak_steps_remaining > 0) {
            accumulate_regime(peak_arrival_distribution, state->peak_steps_remaining - 1, 1.0);
        }
        else {
            accumulate_regime(normal_arrival_distribution, 0, 1.0 - peak_start_prob);
            accumulate_regime(peak_arrival_distribution, peak_duration_steps - 1, peak_start_prob);
        }

        return distr;
    }

    shared_ptr<const ServersSimState> ServersSimEnv::sample_transition_distribution(
        shared_ptr<const ServersSimState> state,
        shared_ptr<const mcts::IntAction> action,
        mcts::RandManager& rand_manager) const
    {
        auto distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double ServersSimEnv::get_reward(
        shared_ptr<const ServersSimState> state,
        shared_ptr<const mcts::IntAction> action,
        shared_ptr<const ServersSimState> observation) const
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

        return -step_cost(observation) / reward_normalisation;
    }

    shared_ptr<const mcts::State> ServersSimEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const mcts::State>(get_initial_state());
    }

    bool ServersSimEnv::is_sink_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_sink_state(static_pointer_cast<const ServersSimState>(state));
    }

    bool ServersSimEnv::is_catastrophic_state_itfc(shared_ptr<const mcts::State> state) const {
        return is_catastrophic_state(static_pointer_cast<const ServersSimState>(state));
    }

    shared_ptr<mcts::ActionVector> ServersSimEnv::get_valid_actions_itfc(shared_ptr<const mcts::State> state) const {
        auto typed_actions = get_valid_actions(static_pointer_cast<const ServersSimState>(state));
        auto actions = make_shared<mcts::ActionVector>();
        for (const auto& typed_action : *typed_actions) {
            actions->push_back(static_pointer_cast<const mcts::Action>(typed_action));
        }
        return actions;
    }

    shared_ptr<mcts::StateDistr> ServersSimEnv::get_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action) const
    {
        auto typed_distr = get_transition_distribution(
            static_pointer_cast<const ServersSimState>(state),
            static_pointer_cast<const mcts::IntAction>(action));
        auto distr = make_shared<mcts::StateDistr>();
        for (const auto& [next_state, probability] : *typed_distr) {
            distr->insert_or_assign(static_pointer_cast<const mcts::State>(next_state), probability);
        }
        return distr;
    }

    shared_ptr<const mcts::State> ServersSimEnv::sample_transition_distribution_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        mcts::RandManager& rand_manager) const
    {
        return static_pointer_cast<const mcts::State>(
            sample_transition_distribution(
                static_pointer_cast<const ServersSimState>(state),
                static_pointer_cast<const mcts::IntAction>(action),
                rand_manager));
    }

    double ServersSimEnv::get_reward_itfc(
        shared_ptr<const mcts::State> state,
        shared_ptr<const mcts::Action> action,
        shared_ptr<const mcts::Observation> observation) const
    {
        shared_ptr<const ServersSimState> typed_observation = nullptr;
        if (observation != nullptr) {
            typed_observation = static_pointer_cast<const ServersSimState>(observation);
        }
        return get_reward(
            static_pointer_cast<const ServersSimState>(state),
            static_pointer_cast<const mcts::IntAction>(action),
            typed_observation);
    }
}
