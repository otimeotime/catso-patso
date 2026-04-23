#include "algorithms/catso/patso_chance_node.h"

#include "algorithms/catso/patso_decision_node.h"
#include "algorithms/catso/patso_manager.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

using namespace std;

namespace mcts {
    PatsoCNode::PatsoCNode(
        shared_ptr<PatsoManager> mcts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const PatsoDNode> parent) :
            CatsoCNode(
                mcts_manager,
                state,
                action,
                decision_depth,
                decision_timestep,
                parent),
            particles(),
            particle_weights()
    {
        mean_value = 0.0;
    }

    int PatsoCNode::find_particle_index(double q_sample) const {
        for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
            if (abs(particles[static_cast<size_t>(i)] - q_sample) <= particle_match_tolerance) {
                return i;
            }
        }
        return -1;
    }

    void PatsoCNode::maybe_merge_particles() {
        PatsoManager& manager = (PatsoManager&) *mcts_manager;
        if (manager.max_particles <= 0) {
            return;
        }

        while (static_cast<int>(particles.size()) > manager.max_particles && particles.size() >= 2u) {
            size_t merge_index = 0;
            double best_gap = abs(particles[1] - particles[0]);
            for (size_t i = 1; i + 1 < particles.size(); ++i) {
                const double gap = abs(particles[i + 1] - particles[i]);
                if (gap < best_gap) {
                    best_gap = gap;
                    merge_index = i;
                }
            }

            const double left_particle = particles[merge_index];
            const double right_particle = particles[merge_index + 1];
            const double left_weight = particle_weights[merge_index];
            const double right_weight = particle_weights[merge_index + 1];
            const double merged_weight = left_weight + right_weight;
            const double merged_particle =
                (left_particle * left_weight + right_particle * right_weight) / merged_weight;

            particles[merge_index] = merged_particle;
            particle_weights[merge_index] = merged_weight;
            particles.erase(particles.begin() + static_cast<ptrdiff_t>(merge_index + 1));
            particle_weights.erase(particle_weights.begin() + static_cast<ptrdiff_t>(merge_index + 1));
        }
    }

    void PatsoCNode::update_distribution(double q_sample) {
        const int particle_index = find_particle_index(q_sample);
        if (particle_index >= 0) {
            particle_weights[static_cast<size_t>(particle_index)] += 1.0;
            return;
        }

        auto insert_it = lower_bound(particles.begin(), particles.end(), q_sample);
        const size_t insert_index = static_cast<size_t>(distance(particles.begin(), insert_it));
        particles.insert(insert_it, q_sample);
        particle_weights.insert(particle_weights.begin() + static_cast<ptrdiff_t>(insert_index), 1.0);
        maybe_merge_particles();
    }

    double PatsoCNode::compute_mean_value() const {
        if (particles.empty() || particle_weights.empty()) {
            return 0.0;
        }

        double sum_weights = 0.0;
        double weighted_sum = 0.0;
        for (size_t i = 0; i < particles.size(); ++i) {
            sum_weights += particle_weights[i];
            weighted_sum += particles[i] * particle_weights[i];
        }

        if (sum_weights <= 0.0) {
            return 0.0;
        }

        return weighted_sum / sum_weights;
    }

    double PatsoCNode::draw_thompson_value() const {
        if (particles.empty() || particle_weights.empty()) {
            return mean_value;
        }

        const vector<double> sampled_weights = sample_dirichlet(particle_weights);
        double sample = 0.0;
        for (size_t i = 0; i < particles.size(); ++i) {
            sample += particles[i] * sampled_weights[i];
        }
        return sample;
    }

    double PatsoCNode::get_cvar_value() const {
        if (particles.empty() || particle_weights.empty()) {
            return mean_value;
        }

        vector<pair<double,double>> value_weight_pairs;
        value_weight_pairs.reserve(particles.size());
        for (size_t i = 0; i < particles.size(); ++i) {
            value_weight_pairs.emplace_back(particles[i], particle_weights[i]);
        }

        PatsoManager& manager = (PatsoManager&) *mcts_manager;
        return compute_lower_tail_cvar(value_weight_pairs, manager.cvar_tau);
    }

    double PatsoCNode::sample_cvar_value() const {
        if (particles.empty() || particle_weights.empty()) {
            return mean_value;
        }

        const vector<double> sampled_weights = sample_dirichlet(particle_weights);
        vector<pair<double,double>> value_weight_pairs;
        value_weight_pairs.reserve(particles.size());
        for (size_t i = 0; i < particles.size(); ++i) {
            value_weight_pairs.emplace_back(particles[i], sampled_weights[i]);
        }

        PatsoManager& manager = (PatsoManager&) *mcts_manager;
        return compute_lower_tail_cvar(value_weight_pairs, manager.cvar_tau);
    }

    shared_ptr<CatsoDNode> PatsoCNode::create_child_node_helper(shared_ptr<const State> observation) const {
        shared_ptr<const State> next_state = static_pointer_cast<const State>(observation);
        return make_shared<PatsoDNode>(
            static_pointer_cast<PatsoManager>(mcts_manager),
            next_state,
            decision_depth + 1,
            decision_timestep + 1,
            static_pointer_cast<const PatsoCNode>(shared_from_this()));
    }

    string PatsoCNode::get_pretty_print_val() const {
        stringstream ss;
        ss << mean_value;
        return ss.str();
    }
}

namespace mcts {
    shared_ptr<PatsoDNode> PatsoCNode::create_child_node(shared_ptr<const State> observation) {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<MctsDNode> new_child = MctsCNode::create_child_node_itfc(obsv_itfc);
        return static_pointer_cast<PatsoDNode>(new_child);
    }

    bool PatsoCNode::has_child_node(shared_ptr<const State> observation) const {
        return MctsCNode::has_child_node_itfc(static_pointer_cast<const Observation>(observation));
    }

    shared_ptr<PatsoDNode> PatsoCNode::get_child_node(shared_ptr<const State> observation) const {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<MctsDNode> new_child = MctsCNode::get_child_node_itfc(obsv_itfc);
        return static_pointer_cast<PatsoDNode>(new_child);
    }

    shared_ptr<MctsDNode> PatsoCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation,
        shared_ptr<const State> next_state) const
    {
        (void)next_state;
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<CatsoDNode> child_node = create_child_node_helper(obsv_itfc);
        return static_pointer_cast<MctsDNode>(child_node);
    }
}
