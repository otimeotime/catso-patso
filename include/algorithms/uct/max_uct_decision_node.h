#pragma once

#include "algorithms/uct/uct_logger.h"
#include "algorithms/uct/uct_manager.h"
#include "mcts_decision_node.h"
#include "mcts_env_context.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace mcts {
    class MaxUctCNode;

    /**
     * Implementation of MaxUCT-style decision nodes in the Mcts schema.
     *
     * This is structurally copied from UctDNode, but inherits directly from MctsDNode so that
     * the backup behaviour can diverge cleanly from standard UCT.
     */
    class MaxUctDNode : public MctsDNode {
        friend MaxUctCNode;
        friend UctLogger;

        protected:
            int num_backups;
            double max_return;
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

            // TODO: Implement max-return backup logic here.
            void backup_max_return();

        public:
            MaxUctDNode(
                std::shared_ptr<UctManager> mcts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const MaxUctCNode> parent=nullptr);

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
            std::shared_ptr<MaxUctCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;
            virtual std::string get_pretty_print_val() const;

        public:
            virtual ~MaxUctDNode() = default;

            std::shared_ptr<MaxUctCNode> create_child_node(std::shared_ptr<const Action> action);
            bool has_child_node(std::shared_ptr<const Action> action) const;
            std::shared_ptr<MaxUctCNode> get_child_node(std::shared_ptr<const Action> action) const;

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
