#pragma once

#include "algorithms/uct/power_uct_manager.h"
#include "algorithms/uct/uct_logger.h"
#include "algorithms/uct/uct_manager.h"
#include "mcts_decision_node.h"
#include "mcts_env_context.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace mcts {
    class PowerUctCNode;

    /**
     * Implementation of MaxUCT-style decision nodes in the Mcts schema.
     *
     * This is structurally copied from UctDNode, but inherits directly from MctsDNode so that
     * the backup behaviour can diverge cleanly from standard UCT.
     */
    class PowerUctDNode : public MctsDNode {
        friend PowerUctCNode;
        friend UctLogger;

        protected:
            int num_backups;
            double power_return;
            std::shared_ptr<ActionVector> actions;
            std::shared_ptr<ActionPrior> policy_prior;

            virtual bool has_prior() const;
            virtual double compute_ucb_term(int num_visits, int child_visits) const;
            virtual void fill_ucb_values(
                std::unordered_map<std::shared_ptr<const Action>, double>& ucb_values,
                MctsEnvContext& ctx) const;
            virtual std::shared_ptr<const Action> select_action_ucb(MctsEnvContext& ctx);
            virtual std::shared_ptr<const Action> select_action_random();

            std::shared_ptr<const Action> recommend_action_best_empirical() const;
            std::shared_ptr<const Action> recommend_action_most_visited() const;

            void backup_power_mean_return();

        public:
            PowerUctDNode(
                std::shared_ptr<PowerUctManager> mcts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const PowerUctCNode> parent=nullptr);

            void visit(MctsEnvContext& ctx);
            std::shared_ptr<const Action> select_action(MctsEnvContext& ctx);
            std::shared_ptr<const Action> recommend_action(MctsEnvContext& ctx) const;
            void backup(
                const std::vector<double>& trial_rewards_before_node,
                const std::vector<double>& trial_rewards_after_node,
                const double trial_cumulative_return_after_node,
                const double trial_cumulative_return,
                MctsEnvContext& ctx);

        protected:
            std::shared_ptr<PowerUctCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;
            virtual std::string get_pretty_print_val() const;

        public:
            virtual ~PowerUctDNode() = default;

            std::shared_ptr<PowerUctCNode> create_child_node(std::shared_ptr<const Action> action);
            bool has_child_node(std::shared_ptr<const Action> action) const;
            std::shared_ptr<PowerUctCNode> get_child_node(std::shared_ptr<const Action> action) const;

            virtual void visit_itfc(MctsEnvContext& ctx);
            virtual std::shared_ptr<const Action> select_action_itfc(MctsEnvContext& ctx);
            virtual std::shared_ptr<const Action> recommend_action_itfc(MctsEnvContext& ctx) const;
            virtual void backup_itfc(
                const std::vector<double>& trial_rewards_before_node,
                const std::vector<double>& trial_rewards_after_node,
                const double trial_cumulative_return_after_node,
                const double trial_cumulative_return,
                MctsEnvContext& ctx);

            virtual std::shared_ptr<MctsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const;
    };
}
