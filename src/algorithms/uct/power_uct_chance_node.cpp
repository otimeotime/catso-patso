#include "algorithms/uct/power_uct_chance_node.h"
#include "algorithms/uct/power_uct_decision_node.h"
#include "helper_templates.h"

using namespace std;

namespace mcts {
    PowerUctCNode::PowerUctCNode(
        shared_ptr<PowerUctManager> mcts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const PowerUctDNode> parent) :
            MctsCNode(
                static_pointer_cast<MctsManager>(mcts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MctsDNode>(parent)),
            num_backups(0),
            power_return(0.0),
            next_state_distr(mcts_manager->mcts_env->get_transition_distribution_itfc(state, action)),
            track_empirical_cvar(parent != nullptr && parent->is_root_node()),
            empirical_return_samples()
    {
    }

    void PowerUctCNode::visit(MctsEnvContext& ctx) {
        MctsCNode::visit_itfc(ctx);
    }

    shared_ptr<const State> PowerUctCNode::sample_observation_random() {
        shared_ptr<const State> sampled_state = helper::sample_from_distribution(*next_state_distr, *mcts_manager);
        if (!has_child_node(sampled_state)) {
            create_child_node(sampled_state);
        }
        return sampled_state;
    }

    shared_ptr<const State> PowerUctCNode::sample_observation(MctsEnvContext& ctx) {
        (void) ctx;
        return sample_observation_random();
    }

    void PowerUctCNode::backup_power_mean_return(const double trial_return_after_node) {
        num_backups++;
        power_return += (trial_return_after_node - power_return) / static_cast<double>(num_backups);
    }

    void PowerUctCNode::backup(
        const vector<double>& trial_rewards_before_node,
        const vector<double>& trial_rewards_after_node,
        const double trial_cumulative_return_after_node,
        const double trial_cumulative_return,
        MctsEnvContext& ctx)
    {
        (void) trial_rewards_before_node;
        (void) trial_cumulative_return;
        (void) ctx;
        (void) trial_rewards_after_node;
        backup_power_mean_return(trial_cumulative_return_after_node);
        if (track_empirical_cvar) {
            empirical_return_samples.push_back(trial_cumulative_return_after_node);
        }
    }

    shared_ptr<PowerUctDNode> PowerUctCNode::create_child_node_helper(shared_ptr<const State> observation) const {
        shared_ptr<const State> next_state = static_pointer_cast<const State>(observation);
        return make_shared<PowerUctDNode>(
            static_pointer_cast<PowerUctManager>(mcts_manager),
            next_state,
            decision_depth + 1,
            decision_timestep + 1,
            static_pointer_cast<const PowerUctCNode>(shared_from_this()));
    }

    string PowerUctCNode::get_pretty_print_val() const {
        stringstream ss;
        ss << power_return;
        return ss.str();
    }

    double PowerUctCNode::get_cvar_value_at(double alpha) const {
        return compute_empirical_lower_tail_cvar_from_samples(empirical_return_samples, alpha);
    }
}

namespace mcts {
    shared_ptr<PowerUctDNode> PowerUctCNode::create_child_node(shared_ptr<const State> observation) {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<MctsDNode> new_child = MctsCNode::create_child_node_itfc(obsv_itfc);
        return static_pointer_cast<PowerUctDNode>(new_child);
    }

    bool PowerUctCNode::has_child_node(shared_ptr<const State> observation) const {
        return MctsCNode::has_child_node_itfc(static_pointer_cast<const Observation>(observation));
    }

    shared_ptr<PowerUctDNode> PowerUctCNode::get_child_node(shared_ptr<const State> observation) const {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<MctsDNode> new_child = MctsCNode::get_child_node_itfc(obsv_itfc);
        return static_pointer_cast<PowerUctDNode>(new_child);
    }
}

namespace mcts {
    void PowerUctCNode::visit_itfc(MctsEnvContext& ctx) {
        visit(ctx);
    }

    shared_ptr<const Observation> PowerUctCNode::sample_observation_itfc(MctsEnvContext& ctx) {
        shared_ptr<const State> obsv = sample_observation(ctx);
        return static_pointer_cast<const Observation>(obsv);
    }

    void PowerUctCNode::backup_itfc(
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

    shared_ptr<MctsDNode> PowerUctCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation,
        shared_ptr<const State> next_state) const
    {
        (void) next_state;
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<PowerUctDNode> child_node = create_child_node_helper(obsv_itfc);
        return static_pointer_cast<MctsDNode>(child_node);
    }
}
