#include "algorithms/uct/max_uct_chance_node.h"

#include "algorithms/uct/max_uct_decision_node.h"
#include "helper_templates.h"

using namespace std;

namespace mcts {
    MaxUctCNode::MaxUctCNode(
        shared_ptr<UctManager> mcts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const MaxUctDNode> parent) :
            MctsCNode(
                static_pointer_cast<MctsManager>(mcts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MctsDNode>(parent)),
            num_backups(0),
            max_return(0.0),
            next_state_distr(mcts_manager->mcts_env->get_transition_distribution_itfc(state, action)),
            track_empirical_cvar(parent != nullptr && parent->is_root_node()),
            empirical_return_samples()
    {
    }

    void MaxUctCNode::visit(MctsEnvContext& ctx) {
        MctsCNode::visit_itfc(ctx);
    }

    shared_ptr<const State> MaxUctCNode::sample_observation_random() {
        shared_ptr<const State> sampled_state = helper::sample_from_distribution(*next_state_distr, *mcts_manager);
        if (!has_child_node(sampled_state)) {
            create_child_node(sampled_state);
        }
        return sampled_state;
    }

    shared_ptr<const State> MaxUctCNode::sample_observation(MctsEnvContext& ctx) {
        (void) ctx;
        return sample_observation_random();
    }

    void MaxUctCNode::backup_max_return(const double immediate_reward) {
        num_backups++;
        max_return = immediate_reward;
        double tmp = 0.0;
        for (const auto& [next_state, prob] : *next_state_distr) {
            if (has_child_node(next_state)) {
                shared_ptr<MaxUctDNode> child_node = get_child_node(next_state);
                tmp += child_node->num_visits * child_node->max_return;
            }
        }
        tmp /= num_visits;
        max_return += tmp;
    }

    void MaxUctCNode::backup(
        const vector<double>& trial_rewards_before_node,
        const vector<double>& trial_rewards_after_node,
        const double trial_cumulative_return_after_node,
        const double trial_cumulative_return,
        MctsEnvContext& ctx)
    {
        (void) trial_rewards_before_node;
        (void) trial_cumulative_return;
        (void) ctx;
        (void)trial_cumulative_return_after_node;
        const double immediate_reward = trial_rewards_after_node.back();
        backup_max_return(immediate_reward);
        if (track_empirical_cvar) {
            empirical_return_samples.push_back(trial_cumulative_return_after_node);
        }
    }

    shared_ptr<MaxUctDNode> MaxUctCNode::create_child_node_helper(shared_ptr<const State> observation) const {
        shared_ptr<const State> next_state = static_pointer_cast<const State>(observation);
        return make_shared<MaxUctDNode>(
            static_pointer_cast<UctManager>(mcts_manager),
            next_state,
            decision_depth + 1,
            decision_timestep + 1,
            static_pointer_cast<const MaxUctCNode>(shared_from_this()));
    }

    string MaxUctCNode::get_pretty_print_val() const {
        stringstream ss;
        ss << max_return;
        return ss.str();
    }

    double MaxUctCNode::get_cvar_value_at(double alpha) const {
        return compute_empirical_lower_tail_cvar_from_samples(empirical_return_samples, alpha);
    }
}

namespace mcts {
    shared_ptr<MaxUctDNode> MaxUctCNode::create_child_node(shared_ptr<const State> observation) {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<MctsDNode> new_child = MctsCNode::create_child_node_itfc(obsv_itfc);
        return static_pointer_cast<MaxUctDNode>(new_child);
    }

    bool MaxUctCNode::has_child_node(shared_ptr<const State> observation) const {
        return MctsCNode::has_child_node_itfc(static_pointer_cast<const Observation>(observation));
    }

    shared_ptr<MaxUctDNode> MaxUctCNode::get_child_node(shared_ptr<const State> observation) const {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<MctsDNode> new_child = MctsCNode::get_child_node_itfc(obsv_itfc);
        return static_pointer_cast<MaxUctDNode>(new_child);
    }
}

namespace mcts {
    void MaxUctCNode::visit_itfc(MctsEnvContext& ctx) {
        visit(ctx);
    }

    shared_ptr<const Observation> MaxUctCNode::sample_observation_itfc(MctsEnvContext& ctx) {
        shared_ptr<const State> obsv = sample_observation(ctx);
        return static_pointer_cast<const Observation>(obsv);
    }

    void MaxUctCNode::backup_itfc(
        const vector<double>& trial_rewards_before_node,
        const vector<double>& trial_rewards_after_node,
        const double trial_cumulative_return_after_node,
        const double trial_cumulative_return,
        MctsEnvContext& ctx)
    {
        backup(
            trial_rewards_before_node,
            trial_rewards_after_node,
            trial_cumulative_return_after_node,
            trial_cumulative_return,
            ctx);
    }

    shared_ptr<MctsDNode> MaxUctCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation,
        shared_ptr<const State> next_state) const
    {
        (void) next_state;
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<MaxUctDNode> child_node = create_child_node_helper(obsv_itfc);
        return static_pointer_cast<MctsDNode>(child_node);
    }
}
