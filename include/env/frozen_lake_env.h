#pragma once

#include "mcts_env.h"
#include "mcts_types.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mcts::exp {
    /**
     * Stochastic Frozen Lake gridworld with terminal goal and hole (trap) states.
     *
     * State is (x,y,t) encoded as Int3TupleState, where t is the timestep (starting at 0).
     * Episode ends when reaching goal, a hole, or when t >= horizon.
     *
     * Transition: intended move succeeds with probability `intended_move_prob`; the remaining probability mass is
     * split equally over the three other compass directions (after wall/bounds handling) and staying put.
     * Reward is sparse: when first reaching the goal at timestep t, reward = gamma^t.
     */
    class FrozenLakeEnv : public mcts::MctsEnv {
        public:
            static constexpr double default_gamma = 0.99;

        private:
            int height;
            int width;
            int horizon;
            double gamma;
            double intended_move_prob;

            std::vector<std::string> grid;
            std::pair<int, int> start_xy;
            std::pair<int, int> goal_xy;

            bool in_bounds(int x, int y) const;
            char cell_at(int x, int y) const;
            bool is_hole_cell(int x, int y) const;
            bool is_goal_cell(int x, int y) const;
            bool is_wall_cell(int x, int y) const;

            std::shared_ptr<const mcts::Int3TupleState> make_next_state_deterministic(
                std::shared_ptr<const mcts::Int3TupleState> state,
                std::shared_ptr<const mcts::StringAction> action) const;

        public:
            /**
             * Construct from an ASCII map.
             *
             * Legend:
             *  - 'S' start (exactly one required)
             *  - 'G' goal  (exactly one required)
             *  - 'H' hole / trap (terminal)
             *  - 'F' frozen (normal)
             *  - 'W' wall (impassable; walking into wall leaves you in place)
             */
            FrozenLakeEnv(
                std::vector<std::string> grid,
                int horizon,
                double gamma = default_gamma,
                double intended_move_prob = 1.0);

            virtual ~FrozenLakeEnv() = default;

            // Typed API
            std::shared_ptr<const mcts::Int3TupleState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const mcts::Int3TupleState> state) const;
            std::shared_ptr<mcts::StringActionVector> get_valid_actions(std::shared_ptr<const mcts::Int3TupleState> state) const;
            std::shared_ptr<mcts::Int3TupleStateDistr> get_transition_distribution(
                std::shared_ptr<const mcts::Int3TupleState> state,
                std::shared_ptr<const mcts::StringAction> action) const;
            std::shared_ptr<const mcts::Int3TupleState> sample_transition_distribution(
                std::shared_ptr<const mcts::Int3TupleState> state,
                std::shared_ptr<const mcts::StringAction> action,
                mcts::RandManager& rand_manager) const;
            double get_reward(
                std::shared_ptr<const mcts::Int3TupleState> state,
                std::shared_ptr<const mcts::StringAction> action,
                std::shared_ptr<const mcts::Int3TupleState> observation = nullptr) const;

            // Interface API
            virtual std::shared_ptr<const mcts::State> get_initial_state_itfc() const;
            virtual bool is_sink_state_itfc(std::shared_ptr<const mcts::State> state) const;
            virtual std::shared_ptr<mcts::ActionVector> get_valid_actions_itfc(std::shared_ptr<const mcts::State> state) const;
            virtual std::shared_ptr<mcts::StateDistr> get_transition_distribution_itfc(
                std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action) const;
            virtual std::shared_ptr<const mcts::State> sample_transition_distribution_itfc(
                std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action,
                mcts::RandManager& rand_manager) const;
            virtual double get_reward_itfc(
                std::shared_ptr<const mcts::State> state,
                std::shared_ptr<const mcts::Action> action,
                std::shared_ptr<const mcts::Observation> observation = nullptr) const;

            int get_horizon() const { return horizon; }
            int get_width() const { return width; }
            int get_height() const { return height; }
            double get_intended_move_prob() const { return intended_move_prob; }
    };
}