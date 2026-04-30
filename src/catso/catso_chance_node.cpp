#include "algorithms/catso/catso_chance_node.h"

#include "algorithms/catso/catso_decision_node.h"
#include "algorithms/catso/catso_manager.h"
#include "helper_templates.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>
#include <vector>

using namespace std;

namespace {
    constexpr double kMinCvarTau = 1e-12;
}

namespace mcts {
    CatsoCNode::CatsoCNode(
        shared_ptr<CatsoManager> mcts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const CatsoDNode> parent) :
            MctsCNode(
                static_pointer_cast<MctsManager>(mcts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MctsDNode>(parent)),
            num_backups(0),
            mean_value(0.0),
            next_state_distr(mcts_manager->mcts_env->get_transition_distribution_itfc(state, action)),
            selected_observation_key(),
            n_atoms(max(1, mcts_manager->n_atoms)),
            q_min(initial_q_min),
            q_max(initial_q_max),
            atoms(static_cast<size_t>(max(1, mcts_manager->n_atoms)), 0.0),
            dirichlet_counts(static_cast<size_t>(max(1, mcts_manager->n_atoms)), 1.0)
    {
        stringstream ss;
        ss << "catso_selected_observation_" << static_cast<const void*>(this);
        selected_observation_key = ss.str();
        reset_atom_grid();
        mean_value = compute_mean_value();
    }

    void CatsoCNode::visit(MctsEnvContext& ctx) {
        MctsCNode::visit_itfc(ctx);
    }

    void CatsoCNode::reset_atom_grid() {
        if (n_atoms <= 0) {
            return;
        }

        if (n_atoms == 1) {
            atoms[0] = q_min;
            return;
        }

        const double delta = (q_max - q_min) / static_cast<double>(n_atoms - 1);
        for (int i = 0; i < n_atoms; ++i) {
            atoms[static_cast<size_t>(i)] = q_min + static_cast<double>(i) * delta;
        }
    }

    int CatsoCNode::nearest_index(const vector<double>& support, double value) const {
        int best_index = 0;
        double best_distance = abs(support[0] - value);
        for (int i = 1; i < static_cast<int>(support.size()); ++i) {
            const double dist = abs(support[static_cast<size_t>(i)] - value);
            if (dist < best_distance) {
                best_distance = dist;
                best_index = i;
            }
        }
        return best_index;
    }

    void CatsoCNode::add_cramer_mass(
        vector<double>& counts,
        const vector<double>& atom_grid,
        double value,
        double weight) const
    {
        if (atom_grid.empty() || counts.size() != atom_grid.size() || weight <= 0.0) {
            return;
        }

        const int grid_size = static_cast<int>(atom_grid.size());
        if (grid_size == 1) {
            counts[0] += weight;
            return;
        }

        const double lo = atom_grid.front();
        const double hi = atom_grid.back();
        if (value <= lo) {
            counts[0] += weight;
            return;
        }
        if (value >= hi) {
            counts[static_cast<size_t>(grid_size - 1)] += weight;
            return;
        }

        const double dz = (hi - lo) / static_cast<double>(grid_size - 1);
        if (dz <= 0.0) {
            counts[0] += weight;
            return;
        }

        int lower = static_cast<int>((value - lo) / dz);
        if (lower < 0) lower = 0;
        if (lower > grid_size - 2) lower = grid_size - 2;
        const int upper = lower + 1;

        const double w_upper = (value - atom_grid[static_cast<size_t>(lower)]) / dz;
        const double w_lower = 1.0 - w_upper;
        counts[static_cast<size_t>(lower)] += weight * w_lower;
        counts[static_cast<size_t>(upper)] += weight * w_upper;
    }

    vector<double> CatsoCNode::sample_dirichlet(const vector<double>& alpha) const {
        if (alpha.empty()) {
            return {};
        }

        vector<double> weights(alpha.size(), 0.0);
        double sum = 0.0;
        for (size_t i = 0; i < alpha.size(); ++i) {
            const double shape = max(alpha[i], 1e-6);
            const double sample = mcts_manager->get_rand_gamma(shape);
            weights[i] = max(sample, 0.0);
            sum += weights[i];
        }

        if (sum <= 0.0) {
            const double uniform_weight = 1.0 / static_cast<double>(weights.size());
            fill(weights.begin(), weights.end(), uniform_weight);
            return weights;
        }

        for (double& weight : weights) {
            weight /= sum;
        }
        return weights;
    }

    double CatsoCNode::compute_lower_tail_cvar(
        const vector<pair<double,double>>& value_weight_pairs,
        double tau) const
    {
        vector<pair<double,double>> sorted_support;
        sorted_support.reserve(value_weight_pairs.size());

        double total_weight = 0.0;
        for (const auto& [value, weight] : value_weight_pairs) {
            if (weight <= 0.0) {
                continue;
            }

            sorted_support.emplace_back(value, weight);
            total_weight += weight;
        }

        if (sorted_support.empty() || total_weight <= 0.0) {
            return 0.0;
        }

        sort(
            sorted_support.begin(),
            sorted_support.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

        const double tau_clamped = max(kMinCvarTau, min(tau, 1.0));
        const double tail_mass = tau_clamped * total_weight;
        double cumulative_weight = 0.0;
        double tail_sum = 0.0;

        for (const auto& [value, weight] : sorted_support) {
            const double remaining_mass = max(0.0, tail_mass - cumulative_weight);
            const double weight_to_take = min(weight, remaining_mass);
            tail_sum += weight_to_take * value;
            cumulative_weight += weight;

            if (cumulative_weight >= tail_mass) {
                break;
            }
        }

        return tail_sum / tail_mass;
    }

    shared_ptr<const State> CatsoCNode::sample_observation_random() {
        shared_ptr<const State> sampled_state = helper::sample_from_distribution(*next_state_distr, *mcts_manager);
        if (!has_child_node(sampled_state)) {
            create_child_node(sampled_state);
        }
        return sampled_state;
    }

    shared_ptr<const State> CatsoCNode::sample_observation(MctsEnvContext& ctx) {
        shared_ptr<const State> sampled_state = sample_observation_random();
        shared_ptr<const Observation> observation = static_pointer_cast<const Observation>(sampled_state);
        ctx.put_value_const(selected_observation_key, observation);
        return sampled_state;
    }

    void CatsoCNode::update_distribution(double q_sample) {
        if (atoms.empty()) {
            return;
        }

        if (q_sample >= atoms.front() - match_tolerance && q_sample <= atoms.back() + match_tolerance) {
            add_cramer_mass(dirichlet_counts, atoms, q_sample, 1.0);
            return;
        }

        const double new_min = min(atoms.front(), q_sample);
        const double new_max = max(atoms.back(), q_sample);

        vector<double> new_atoms(static_cast<size_t>(n_atoms), 0.0);
        if (n_atoms == 1) {
            new_atoms[0] = new_min;
        } else {
            const double delta = (new_max - new_min) / static_cast<double>(n_atoms - 1);
            for (int i = 0; i < n_atoms; ++i) {
                new_atoms[static_cast<size_t>(i)] = new_min + static_cast<double>(i) * delta;
            }
        }

        // Re-bin existing posterior counts (which already include the +1 prior added at
        // construction) onto the new grid via Cramér projection. Do NOT re-inject the
        // prior here -- doing so would inflate the pseudo-count by N every time the
        // grid expands.
        vector<double> new_counts(static_cast<size_t>(n_atoms), 0.0);
        for (int i = 0; i < n_atoms; ++i) {
            add_cramer_mass(
                new_counts,
                new_atoms,
                atoms[static_cast<size_t>(i)],
                dirichlet_counts[static_cast<size_t>(i)]);
        }

        add_cramer_mass(new_counts, new_atoms, q_sample, 1.0);

        q_min = new_min;
        q_max = new_max;
        atoms = move(new_atoms);
        dirichlet_counts = move(new_counts);
    }

    double CatsoCNode::compute_mean_value() const {
        if (atoms.empty() || dirichlet_counts.empty()) {
            return 0.0;
        }

        double sum_counts = 0.0;
        double weighted_sum = 0.0;
        for (size_t i = 0; i < atoms.size(); ++i) {
            sum_counts += dirichlet_counts[i];
            weighted_sum += atoms[i] * dirichlet_counts[i];
        }

        if (sum_counts <= 0.0) {
            return 0.0;
        }

        return weighted_sum / sum_counts;
    }

    double CatsoCNode::draw_thompson_value() const {
        if (atoms.empty() || dirichlet_counts.empty()) {
            return mean_value;
        }

        const vector<double> sampled_weights = sample_dirichlet(dirichlet_counts);
        double sample = 0.0;
        for (size_t i = 0; i < atoms.size(); ++i) {
            sample += atoms[i] * sampled_weights[i];
        }
        return sample;
    }

    void CatsoCNode::backup(
        const vector<double>& trial_rewards_before_node,
        const vector<double>& trial_rewards_after_node,
        const double trial_cumulative_return_after_node,
        const double trial_cumulative_return,
        MctsEnvContext& ctx)
    {
        (void)trial_rewards_before_node;
        (void)trial_rewards_after_node;
        (void)trial_cumulative_return_after_node;
        (void)trial_cumulative_return;

        num_backups++;
        shared_ptr<const Observation> observation = ctx.get_value_ptr_const<Observation>(selected_observation_key);
        const double immediate_reward = mcts_manager->mcts_env->get_reward_itfc(state, action, observation);

        double child_value = 0.0;
        shared_ptr<const State> sampled_state = static_pointer_cast<const State>(observation);
        if (has_child_node(sampled_state)) {
            shared_ptr<CatsoDNode> child = get_child_node(sampled_state);
            child_value = child->value_estimate;
        }

        update_distribution(immediate_reward + child_value);
        mean_value = compute_mean_value();
    }

    double CatsoCNode::get_mean_value() const {
        return mean_value;
    }

    double CatsoCNode::get_cvar_value() const {
        if (atoms.empty() || dirichlet_counts.empty()) {
            return mean_value;
        }

        vector<pair<double,double>> value_weight_pairs;
        value_weight_pairs.reserve(atoms.size());
        for (size_t i = 0; i < atoms.size(); ++i) {
            value_weight_pairs.emplace_back(atoms[i], dirichlet_counts[i]);
        }

        CatsoManager& manager = (CatsoManager&) *mcts_manager;
        return compute_lower_tail_cvar(value_weight_pairs, manager.cvar_tau);
    }

    double CatsoCNode::sample_cvar_value() const {
        if (atoms.empty() || dirichlet_counts.empty()) {
            return mean_value;
        }

        const vector<double> sampled_weights = sample_dirichlet(dirichlet_counts);
        vector<pair<double,double>> value_weight_pairs;
        value_weight_pairs.reserve(atoms.size());
        for (size_t i = 0; i < atoms.size(); ++i) {
            value_weight_pairs.emplace_back(atoms[i], sampled_weights[i]);
        }

        CatsoManager& manager = (CatsoManager&) *mcts_manager;
        return compute_lower_tail_cvar(value_weight_pairs, manager.cvar_tau);
    }

    double CatsoCNode::sample_thompson_value() const {
        return draw_thompson_value();
    }

    shared_ptr<CatsoDNode> CatsoCNode::create_child_node_helper(shared_ptr<const State> observation) const {
        shared_ptr<const State> next_state = static_pointer_cast<const State>(observation);
        return make_shared<CatsoDNode>(
            static_pointer_cast<CatsoManager>(mcts_manager),
            next_state,
            decision_depth + 1,
            decision_timestep + 1,
            static_pointer_cast<const CatsoCNode>(shared_from_this()));
    }

    string CatsoCNode::get_pretty_print_val() const {
        stringstream ss;
        ss << mean_value;
        return ss.str();
    }
}

namespace mcts {
    shared_ptr<CatsoDNode> CatsoCNode::create_child_node(shared_ptr<const State> observation) {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<MctsDNode> new_child = MctsCNode::create_child_node_itfc(obsv_itfc);
        return static_pointer_cast<CatsoDNode>(new_child);
    }

    bool CatsoCNode::has_child_node(shared_ptr<const State> observation) const {
        return MctsCNode::has_child_node_itfc(static_pointer_cast<const Observation>(observation));
    }

    shared_ptr<CatsoDNode> CatsoCNode::get_child_node(shared_ptr<const State> observation) const {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<MctsDNode> new_child = MctsCNode::get_child_node_itfc(obsv_itfc);
        return static_pointer_cast<CatsoDNode>(new_child);
    }
}

namespace mcts {
    void CatsoCNode::visit_itfc(MctsEnvContext& ctx) {
        MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Observation> CatsoCNode::sample_observation_itfc(MctsEnvContext& ctx) {
        MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
        shared_ptr<const State> obsv = sample_observation(ctx_itfc);
        return static_pointer_cast<const Observation>(obsv);
    }

    void CatsoCNode::backup_itfc(
        const vector<double>& trial_rewards_before_node,
        const vector<double>& trial_rewards_after_node,
        const double trial_cumulative_return_after_node,
        const double trial_cumulative_return,
        MctsEnvContext& ctx)
    {
        MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
        backup(
            trial_rewards_before_node,
            trial_rewards_after_node,
            trial_cumulative_return_after_node,
            trial_cumulative_return,
            ctx_itfc);
    }

    shared_ptr<MctsDNode> CatsoCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation,
        shared_ptr<const State> next_state) const
    {
        (void)next_state;
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<CatsoDNode> child_node = create_child_node_helper(obsv_itfc);
        return static_pointer_cast<MctsDNode>(child_node);
    }
}
