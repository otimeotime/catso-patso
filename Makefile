CXX = g++



#####
# Defining targets, sources and flags
#####

SRC_DIR = src
BIN_DIR = bin

COMMON_SOURCES = $(wildcard src/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/common/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/est/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/ments/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/ments/dents/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/ments/rents/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/ments/tents/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/varde/*.cpp)
COMMON_SOURCES += $(wildcard src/algorithms/uct/*.cpp)
COMMON_SOURCES += $(wildcard src/catso/*.cpp)
COMMON_SOURCES += $(wildcard src/distributions/*.cpp)
COMMON_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(COMMON_SOURCES))

# Shared environment implementations used by experiment binaries
ENV_SOURCES = $(wildcard src/env/*.cpp)
ENV_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(ENV_SOURCES))

# Parameter tuning main for FrozenLake
TUNE_SOURCES = src/exp/tune_frozenlake.cpp
TUNE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(TUNE_SOURCES))

# Parameter tuning main for Sailing
TUNE_SAILING_SOURCES = src/exp/tune_sailing.cpp
TUNE_SAILING_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(TUNE_SAILING_SOURCES))

# Parameter tuning main for Taxi
TUNE_TAXI_SOURCES = src/exp/tune_taxi.cpp
TUNE_TAXI_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(TUNE_TAXI_SOURCES))

# Parameter tuning main for Synthetic Tree
TUNE_STREE_SOURCES = src/exp/tune_stree.cpp
TUNE_STREE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(TUNE_STREE_SOURCES))

# Parameter tuning main for BettingGame
TUNE_BETTINGGAME_SOURCES = src/exp/tune_bettinggame.cpp
TUNE_BETTINGGAME_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(TUNE_BETTINGGAME_SOURCES))

# FrozenLake experiment with tuned parameters
RUN_SOURCES = src/exp/run_frozenlake.cpp
RUN_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_SOURCES))

# Sailing experiment with tuned parameters
RUN_SAILING_SOURCES = src/exp/run_sailing.cpp
RUN_SAILING_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_SAILING_SOURCES))

# Taxi experiment with tuned parameters
RUN_TAXI_SOURCES = src/exp/run_taxi.cpp
RUN_TAXI_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_TAXI_SOURCES))

# BettingGame experiment
RUN_BETTINGGAME_SOURCES = src/exp/run_bettinggame.cpp
RUN_BETTINGGAME_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_BETTINGGAME_SOURCES))

# BettingGame single-config evaluator (driven by tune/tune.py)
EVAL_BETTINGGAME_SOURCES = src/exp/eval_bettinggame.cpp
EVAL_BETTINGGAME_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(EVAL_BETTINGGAME_SOURCES))

# Autonomous Vehicle single-config evaluator
EVAL_AUTONOMOUS_VEHICLE_SOURCES = src/exp/eval_autonomous_vehicle.cpp
EVAL_AUTONOMOUS_VEHICLE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(EVAL_AUTONOMOUS_VEHICLE_SOURCES))

# Guarded Maze single-config evaluator
EVAL_GUARDED_MAZE_SOURCES = src/exp/eval_guarded_maze.cpp
EVAL_GUARDED_MAZE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(EVAL_GUARDED_MAZE_SOURCES))

# Lunar Lander single-config evaluator
EVAL_LUNAR_LANDER_SOURCES = src/exp/eval_lunar_lander.cpp
EVAL_LUNAR_LANDER_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(EVAL_LUNAR_LANDER_SOURCES))

# Risky Shortcut Gridworld experiment
RUN_RISKY_SHORTCUT_GRIDWORLD_SOURCES = src/exp/run_risky_shortcut_gridworld.cpp
RUN_RISKY_SHORTCUT_GRIDWORLD_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_RISKY_SHORTCUT_GRIDWORLD_SOURCES))

# Autonomous Vehicle experiment
RUN_AUTONOMOUS_VEHICLE_SOURCES = src/exp/run_autonomous_vehicle.cpp
RUN_AUTONOMOUS_VEHICLE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_AUTONOMOUS_VEHICLE_SOURCES))

# Guarded Maze experiment
RUN_GUARDED_MAZE_SOURCES = src/exp/run_guarded_maze.cpp
RUN_GUARDED_MAZE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_GUARDED_MAZE_SOURCES))

# Thin Ice Frozen Lake Plus experiment
RUN_THIN_ICE_FROZEN_LAKE_PLUS_SOURCES = src/exp/run_thin_ice_frozen_lake_plus.cpp
RUN_THIN_ICE_FROZEN_LAKE_PLUS_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_THIN_ICE_FROZEN_LAKE_PLUS_SOURCES))

# Overflow Queue experiment
RUN_OVERFLOW_QUEUE_SOURCES = src/exp/run_overflow_queue.cpp
RUN_OVERFLOW_QUEUE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_OVERFLOW_QUEUE_SOURCES))

# Gambler Jackpot experiment
RUN_GAMBLER_JACKPOT_SOURCES = src/exp/run_gambler_jackpot.cpp
RUN_GAMBLER_JACKPOT_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_GAMBLER_JACKPOT_SOURCES))

# Risky Ladder experiment
RUN_RISKY_LADDER_SOURCES = src/exp/run_risky_ladder.cpp
RUN_RISKY_LADDER_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_RISKY_LADDER_SOURCES))

# Two Path SSP Deceptive experiment
RUN_TWO_PATH_SSP_DECEPTIVE_SOURCES = src/exp/run_two_path_ssp_deceptive.cpp
RUN_TWO_PATH_SSP_DECEPTIVE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_TWO_PATH_SSP_DECEPTIVE_SOURCES))

# Laser Tag Safe Grid experiment
RUN_LASER_TAG_SAFE_GRID_SOURCES = src/exp/run_laser_tag_safe_grid.cpp
RUN_LASER_TAG_SAFE_GRID_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_LASER_TAG_SAFE_GRID_SOURCES))

# River Swim Stochastic experiment
RUN_RIVER_SWIM_STOCHASTIC_SOURCES = src/exp/run_river_swim_stochastic.cpp
RUN_RIVER_SWIM_STOCHASTIC_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_RIVER_SWIM_STOCHASTIC_SOURCES))

# Two Level Risky Treasure experiment
RUN_TWO_LEVEL_RISKY_TREASURE_SOURCES = src/exp/run_two_level_risky_treasure.cpp
RUN_TWO_LEVEL_RISKY_TREASURE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_TWO_LEVEL_RISKY_TREASURE_SOURCES))

# Synthetic Tree experiment with tuned parameters
RUN_STREE_SOURCES = src/exp/run_stree.cpp
RUN_STREE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(RUN_STREE_SOURCES))

# VarDE debug main
VarDE_DBG_SOURCES = src/exp/debug_varde.cpp
VarDE_DBG_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(VarDE_DBG_SOURCES))
# DENTS debug main
DENTS_DBG_SOURCES = src/exp/debug_dents.cpp
DENTS_DBG_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(DENTS_DBG_SOURCES))
# VarDE Sailing debug main
VarDE_DBG_SAILING_SOURCES = src/exp/debug_varde_sailing.cpp
VarDE_DBG_SAILING_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(VarDE_DBG_SAILING_SOURCES))
# VarDE Taxi debug main
VarDE_DBG_TAXI_SOURCES = src/exp/debug_varde_taxi.cpp
VarDE_DBG_TAXI_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(VarDE_DBG_TAXI_SOURCES))
# DENTS Sailing debug main
DENTS_DBG_SAILING_SOURCES = src/exp/debug_dents_sailing.cpp
DENTS_DBG_SAILING_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(DENTS_DBG_SAILING_SOURCES))
# DENTS Taxi debug main
DENTS_DBG_TAXI_SOURCES = src/exp/debug_dents_taxi.cpp
DENTS_DBG_TAXI_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(DENTS_DBG_TAXI_SOURCES))

# VarDE Synthetic Tree debug main
VarDE_DBG_STREE_SOURCES = src/exp/debug_varde_stree.cpp
VarDE_DBG_STREE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(VarDE_DBG_STREE_SOURCES))
# DENTS Synthetic Tree debug main
DENTS_DBG_STREE_SOURCES = src/exp/debug_dents_stree.cpp
DENTS_DBG_STREE_OBJECTS = $(patsubst src/%.cpp, bin/src/%.o, $(DENTS_DBG_STREE_SOURCES))

INCLUDES = -Iinclude/ -Isrc/ -I.

CPPFLAGS = $(INCLUDES) -Wall -std=c++20
CPPFLAGS_DEBUG = -g
DEPFLAGS = -MMD -MP

LDFLAGS = -lpthread

TARGET_MCTS = mcts
TARGET_MCTS_TUNE = mcts-tune-frozenlake
TARGET_MCTS_TUNE_SAILING = mcts-tune-sailing
TARGET_MCTS_TUNE_TAXI = mcts-tune-taxi
TARGET_MCTS_TUNE_STREE = mcts-tune-stree
TARGET_MCTS_TUNE_BETTINGGAME = mcts-tune-bettinggame
TARGET_MCTS_EVAL_BETTINGGAME = mcts-eval-bettinggame
TARGET_MCTS_EVAL_AUTONOMOUS_VEHICLE = mcts-eval-autonomous-vehicle
TARGET_MCTS_EVAL_GUARDED_MAZE = mcts-eval-guarded-maze
TARGET_MCTS_EVAL_LUNAR_LANDER = mcts-eval-lunar-lander
TARGET_MCTS_RUN = mcts-run-frozenlake
TARGET_MCTS_RUN_SAILING = mcts-run-sailing
TARGET_MCTS_RUN_TAXI = mcts-run-taxi
TARGET_MCTS_RUN_BETTINGGAME = mcts-run-bettinggame
TARGET_MCTS_RUN_RISKY_SHORTCUT_GRIDWORLD = mcts-run-risky-shortcut-gridworld
TARGET_MCTS_RUN_AUTONOMOUS_VEHICLE = mcts-run-autonomous-vehicle
TARGET_MCTS_RUN_GUARDED_MAZE = mcts-run-guarded-maze
TARGET_MCTS_RUN_THIN_ICE_FROZEN_LAKE_PLUS = mcts-run-thin-ice-frozen-lake-plus
TARGET_MCTS_RUN_OVERFLOW_QUEUE = mcts-run-overflow-queue
TARGET_MCTS_RUN_GAMBLER_JACKPOT = mcts-run-gambler-jackpot
TARGET_MCTS_RUN_RISKY_LADDER = mcts-run-risky-ladder
TARGET_MCTS_RUN_TWO_PATH_SSP_DECEPTIVE = mcts-run-two-path-ssp-deceptive
TARGET_MCTS_RUN_LASER_TAG_SAFE_GRID = mcts-run-laser-tag-safe-grid
TARGET_MCTS_RUN_RIVER_SWIM_STOCHASTIC = mcts-run-river-swim-stochastic
TARGET_MCTS_RUN_TWO_LEVEL_RISKY_TREASURE = mcts-run-two-level-risky-treasure
TARGET_MCTS_RUN_STREE = mcts-run-stree
TARGET_MCTS_VarDE_DBG = mcts-debug-varde
TARGET_MCTS_DENTS_DBG = mcts-debug-dents
TARGET_MCTS_VarDE_DBG_SAILING = mcts-debug-varde-sailing
TARGET_MCTS_DENTS_DBG_SAILING = mcts-debug-dents-sailing
TARGET_MCTS_VarDE_DBG_TAXI = mcts-debug-varde-taxi
TARGET_MCTS_DENTS_DBG_TAXI = mcts-debug-dents-taxi
TARGET_MCTS_VarDE_DBG_STREE = mcts-debug-varde-stree
TARGET_MCTS_DENTS_DBG_STREE = mcts-debug-dents-stree



#####
# Default target
#####

# Default, build common objects
all: $(TARGET_MCTS)

DEPFILES = \
	$(COMMON_OBJECTS:.o=.d) \
	$(ENV_OBJECTS:.o=.d) \
	$(TUNE_OBJECTS:.o=.d) \
	$(TUNE_SAILING_OBJECTS:.o=.d) \
	$(TUNE_TAXI_OBJECTS:.o=.d) \
	$(TUNE_STREE_OBJECTS:.o=.d) \
	$(TUNE_BETTINGGAME_OBJECTS:.o=.d) \
	$(EVAL_BETTINGGAME_OBJECTS:.o=.d) \
	$(EVAL_AUTONOMOUS_VEHICLE_OBJECTS:.o=.d) \
	$(EVAL_GUARDED_MAZE_OBJECTS:.o=.d) \
	$(EVAL_LUNAR_LANDER_OBJECTS:.o=.d) \
	$(RUN_OBJECTS:.o=.d) \
	$(RUN_SAILING_OBJECTS:.o=.d) \
	$(RUN_TAXI_OBJECTS:.o=.d) \
	$(RUN_BETTINGGAME_OBJECTS:.o=.d) \
	$(RUN_RISKY_SHORTCUT_GRIDWORLD_OBJECTS:.o=.d) \
	$(RUN_AUTONOMOUS_VEHICLE_OBJECTS:.o=.d) \
	$(RUN_GUARDED_MAZE_OBJECTS:.o=.d) \
	$(RUN_THIN_ICE_FROZEN_LAKE_PLUS_OBJECTS:.o=.d) \
	$(RUN_OVERFLOW_QUEUE_OBJECTS:.o=.d) \
	$(RUN_GAMBLER_JACKPOT_OBJECTS:.o=.d) \
	$(RUN_RISKY_LADDER_OBJECTS:.o=.d) \
	$(RUN_TWO_PATH_SSP_DECEPTIVE_OBJECTS:.o=.d) \
	$(RUN_LASER_TAG_SAFE_GRID_OBJECTS:.o=.d) \
	$(RUN_RIVER_SWIM_STOCHASTIC_OBJECTS:.o=.d) \
	$(RUN_TWO_LEVEL_RISKY_TREASURE_OBJECTS:.o=.d) \
	$(RUN_STREE_OBJECTS:.o=.d) \
	$(VarDE_DBG_OBJECTS:.o=.d) \
	$(DENTS_DBG_OBJECTS:.o=.d) \
	$(VarDE_DBG_SAILING_OBJECTS:.o=.d) \
	$(VarDE_DBG_TAXI_OBJECTS:.o=.d) \
	$(DENTS_DBG_SAILING_OBJECTS:.o=.d) \
	$(DENTS_DBG_TAXI_OBJECTS:.o=.d) \
	$(VarDE_DBG_STREE_OBJECTS:.o=.d) \
	$(DENTS_DBG_STREE_OBJECTS:.o=.d)

-include $(DEPFILES)



#####
# (Custom) Rules to build object files
# Adapted from: https://stackoverflow.com/questions/41568508/makefile-compile-multiple-c-file-at-once/41924169#41924169
# N.B. A rule of the form "some_dir/%.o: bin/some_dir/%.cpp" for some reason makes make recompile all files everytime
#####

# Required to enable use of $$
.SECONDEXPANSION:

## General rule formula for building object files
## N.B. $(@D) gets the directory of the file
#$(OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
#	@mkdir -p $(@D)
#	$(CXX) $(CPPFLAGS) -c -o $@ $<

# Build object files rule
$(COMMON_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build environment object files rule
$(ENV_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build tuner object files rule
$(TUNE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build sailing tuner object files rule
$(TUNE_SAILING_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build taxi tuner object files rule
$(TUNE_TAXI_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build synthetic tree tuner object files rule
$(TUNE_STREE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build betting game tuner object files rule
$(TUNE_BETTINGGAME_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build betting game eval object files rule
$(EVAL_BETTINGGAME_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build autonomous vehicle eval object files rule
$(EVAL_AUTONOMOUS_VEHICLE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build guarded maze eval object files rule
$(EVAL_GUARDED_MAZE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build lunar lander eval object files rule
$(EVAL_LUNAR_LANDER_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build run object files rule
$(RUN_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build sailing run object files rule
$(RUN_SAILING_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build taxi run object files rule
$(RUN_TAXI_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build betting game run object files rule
$(RUN_BETTINGGAME_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build risky shortcut gridworld run object files rule
$(RUN_RISKY_SHORTCUT_GRIDWORLD_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build autonomous vehicle run object files rule
$(RUN_AUTONOMOUS_VEHICLE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build guarded maze run object files rule
$(RUN_GUARDED_MAZE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build thin ice frozen lake plus run object files rule
$(RUN_THIN_ICE_FROZEN_LAKE_PLUS_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build overflow queue run object files rule
$(RUN_OVERFLOW_QUEUE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build gambler jackpot run object files rule
$(RUN_GAMBLER_JACKPOT_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build risky ladder run object files rule
$(RUN_RISKY_LADDER_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build two path SSP deceptive run object files rule
$(RUN_TWO_PATH_SSP_DECEPTIVE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build laser tag safe grid run object files rule
$(RUN_LASER_TAG_SAFE_GRID_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build river swim stochastic run object files rule
$(RUN_RIVER_SWIM_STOCHASTIC_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build two level risky treasure run object files rule
$(RUN_TWO_LEVEL_RISKY_TREASURE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build synthetic tree run object files rule
$(RUN_STREE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build VarDE debug object files rule
$(VarDE_DBG_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build DENTS debug object files rule
$(DENTS_DBG_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build VarDE Sailing debug object files rule
$(VarDE_DBG_SAILING_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build VarDE Taxi debug object files rule
$(VarDE_DBG_TAXI_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build DENTS Sailing debug object files rule
$(DENTS_DBG_SAILING_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build DENTS Taxi debug object files rule
$(DENTS_DBG_TAXI_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build VarDE Synthetic Tree debug object files rule
$(VarDE_DBG_STREE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<

# Build DENTS Synthetic Tree debug object files rule
$(DENTS_DBG_STREE_OBJECTS): $$(patsubst $(BIN_DIR)/%.o, %.cpp, $$@)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) -c -o $@ $<



#####
# Targets
#####

# Compiling 'mcts' just builds objects for now (to be used as prereq basically)
$(TARGET_MCTS): $(COMMON_OBJECTS)

# Build FrozenLake tuner
$(TARGET_MCTS_TUNE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(TUNE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Sailing tuner
$(TARGET_MCTS_TUNE_SAILING): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(TUNE_SAILING_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Taxi tuner
$(TARGET_MCTS_TUNE_TAXI): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(TUNE_TAXI_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Synthetic Tree tuner
$(TARGET_MCTS_TUNE_STREE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(TUNE_STREE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build BettingGame tuner
$(TARGET_MCTS_TUNE_BETTINGGAME): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(TUNE_BETTINGGAME_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build BettingGame single-config evaluator (driven by tune/tune.py)
$(TARGET_MCTS_EVAL_BETTINGGAME): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(EVAL_BETTINGGAME_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Autonomous Vehicle single-config evaluator
$(TARGET_MCTS_EVAL_AUTONOMOUS_VEHICLE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(EVAL_AUTONOMOUS_VEHICLE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Guarded Maze single-config evaluator
$(TARGET_MCTS_EVAL_GUARDED_MAZE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(EVAL_GUARDED_MAZE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Lunar Lander single-config evaluator
$(TARGET_MCTS_EVAL_LUNAR_LANDER): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(EVAL_LUNAR_LANDER_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build FrozenLake experiment with tuned parameters
$(TARGET_MCTS_RUN): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Sailing experiment with tuned parameters
$(TARGET_MCTS_RUN_SAILING): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_SAILING_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Taxi experiment with tuned parameters
$(TARGET_MCTS_RUN_TAXI): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_TAXI_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build BettingGame experiment
$(TARGET_MCTS_RUN_BETTINGGAME): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_BETTINGGAME_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Risky Shortcut Gridworld experiment
$(TARGET_MCTS_RUN_RISKY_SHORTCUT_GRIDWORLD): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_RISKY_SHORTCUT_GRIDWORLD_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Autonomous Vehicle experiment
$(TARGET_MCTS_RUN_AUTONOMOUS_VEHICLE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_AUTONOMOUS_VEHICLE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Guarded Maze experiment
$(TARGET_MCTS_RUN_GUARDED_MAZE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_GUARDED_MAZE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Thin Ice Frozen Lake Plus experiment
$(TARGET_MCTS_RUN_THIN_ICE_FROZEN_LAKE_PLUS): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_THIN_ICE_FROZEN_LAKE_PLUS_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Overflow Queue experiment
$(TARGET_MCTS_RUN_OVERFLOW_QUEUE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_OVERFLOW_QUEUE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Gambler Jackpot experiment
$(TARGET_MCTS_RUN_GAMBLER_JACKPOT): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_GAMBLER_JACKPOT_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Risky Ladder experiment
$(TARGET_MCTS_RUN_RISKY_LADDER): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_RISKY_LADDER_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Two Path SSP Deceptive experiment
$(TARGET_MCTS_RUN_TWO_PATH_SSP_DECEPTIVE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_TWO_PATH_SSP_DECEPTIVE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Laser Tag Safe Grid experiment
$(TARGET_MCTS_RUN_LASER_TAG_SAFE_GRID): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_LASER_TAG_SAFE_GRID_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build River Swim Stochastic experiment
$(TARGET_MCTS_RUN_RIVER_SWIM_STOCHASTIC): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_RIVER_SWIM_STOCHASTIC_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Two Level Risky Treasure experiment
$(TARGET_MCTS_RUN_TWO_LEVEL_RISKY_TREASURE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_TWO_LEVEL_RISKY_TREASURE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build Synthetic Tree experiment with tuned parameters
$(TARGET_MCTS_RUN_STREE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(RUN_STREE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build VarDE debug binary
$(TARGET_MCTS_VarDE_DBG): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(VarDE_DBG_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build DENTS debug binary
$(TARGET_MCTS_DENTS_DBG): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(DENTS_DBG_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build VarDE Sailing debug binary
$(TARGET_MCTS_VarDE_DBG_SAILING): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(VarDE_DBG_SAILING_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build VarDE Taxi debug binary
$(TARGET_MCTS_VarDE_DBG_TAXI): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(VarDE_DBG_TAXI_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build DENTS Sailing debug binary
$(TARGET_MCTS_DENTS_DBG_SAILING): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(DENTS_DBG_SAILING_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build DENTS Taxi debug binary
$(TARGET_MCTS_DENTS_DBG_TAXI): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(DENTS_DBG_TAXI_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build VarDE Synthetic Tree debug binary
$(TARGET_MCTS_VarDE_DBG_STREE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(VarDE_DBG_STREE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

# Build DENTS Synthetic Tree debug binary
$(TARGET_MCTS_DENTS_DBG_STREE): $(COMMON_OBJECTS) $(ENV_OBJECTS) $(DENTS_DBG_STREE_OBJECTS)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)




#####
# Clean
#####

# Clean up compiled files
# "[X] && Y || Z" is bash for if X then Y else Z
# [ -e <filename> ] checks if filename exists
# : is a no-op
clean:
	@rm -rf $(BIN_DIR) > /dev/null 2> /dev/null
	@[ -f $(TARGET_MCTS_TUNE) ] && @rm $(TARGET_MCTS_TUNE) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_TUNE_SAILING) ] && @rm $(TARGET_MCTS_TUNE_SAILING) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_TUNE_TAXI) ] && @rm $(TARGET_MCTS_TUNE_TAXI) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_TUNE_STREE) ] && @rm $(TARGET_MCTS_TUNE_STREE) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_TUNE_BETTINGGAME) ] && @rm $(TARGET_MCTS_TUNE_BETTINGGAME) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_EVAL_BETTINGGAME) ] && @rm $(TARGET_MCTS_EVAL_BETTINGGAME) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_EVAL_AUTONOMOUS_VEHICLE) ] && @rm $(TARGET_MCTS_EVAL_AUTONOMOUS_VEHICLE) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_EVAL_GUARDED_MAZE) ] && @rm $(TARGET_MCTS_EVAL_GUARDED_MAZE) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_EVAL_LUNAR_LANDER) ] && @rm $(TARGET_MCTS_EVAL_LUNAR_LANDER) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN) ] && @rm $(TARGET_MCTS_RUN) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_SAILING) ] && @rm $(TARGET_MCTS_RUN_SAILING) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_TAXI) ] && @rm $(TARGET_MCTS_RUN_TAXI) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_BETTINGGAME) ] && @rm $(TARGET_MCTS_RUN_BETTINGGAME) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_RISKY_SHORTCUT_GRIDWORLD) ] && @rm $(TARGET_MCTS_RUN_RISKY_SHORTCUT_GRIDWORLD) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_AUTONOMOUS_VEHICLE) ] && @rm $(TARGET_MCTS_RUN_AUTONOMOUS_VEHICLE) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_GUARDED_MAZE) ] && @rm $(TARGET_MCTS_RUN_GUARDED_MAZE) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_THIN_ICE_FROZEN_LAKE_PLUS) ] && @rm $(TARGET_MCTS_RUN_THIN_ICE_FROZEN_LAKE_PLUS) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_OVERFLOW_QUEUE) ] && @rm $(TARGET_MCTS_RUN_OVERFLOW_QUEUE) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_GAMBLER_JACKPOT) ] && @rm $(TARGET_MCTS_RUN_GAMBLER_JACKPOT) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_RISKY_LADDER) ] && @rm $(TARGET_MCTS_RUN_RISKY_LADDER) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_TWO_PATH_SSP_DECEPTIVE) ] && @rm $(TARGET_MCTS_RUN_TWO_PATH_SSP_DECEPTIVE) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_LASER_TAG_SAFE_GRID) ] && @rm $(TARGET_MCTS_RUN_LASER_TAG_SAFE_GRID) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_RIVER_SWIM_STOCHASTIC) ] && @rm $(TARGET_MCTS_RUN_RIVER_SWIM_STOCHASTIC) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_TWO_LEVEL_RISKY_TREASURE) ] && @rm $(TARGET_MCTS_RUN_TWO_LEVEL_RISKY_TREASURE) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_RUN_STREE) ] && @rm $(TARGET_MCTS_RUN_STREE) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_VarDE_DBG) ] && @rm $(TARGET_MCTS_VarDE_DBG) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_DENTS_DBG) ] && @rm $(TARGET_MCTS_DENTS_DBG) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_VarDE_DBG_SAILING) ] && @rm $(TARGET_MCTS_VarDE_DBG_SAILING) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_DENTS_DBG_SAILING) ] && @rm $(TARGET_MCTS_DENTS_DBG_SAILING) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_VarDE_DBG_STREE) ] && @rm $(TARGET_MCTS_VarDE_DBG_STREE) > /dev/null 2> /dev/null || :
	@[ -f $(TARGET_MCTS_DENTS_DBG_STREE) ] && @rm $(TARGET_MCTS_DENTS_DBG_STREE) > /dev/null 2> /dev/null || :


#####
# Phony targets, so make knows when a target isn't producing a corresponding output file of same name
#####
.PHONY: clean $(TARGET_MCTS) $(TARGET_MCTS_TUNE) $(TARGET_MCTS_TUNE_SAILING) $(TARGET_MCTS_TUNE_TAXI) $(TARGET_MCTS_TUNE_STREE) $(TARGET_MCTS_TUNE_BETTINGGAME) $(TARGET_MCTS_EVAL_BETTINGGAME) $(TARGET_MCTS_EVAL_AUTONOMOUS_VEHICLE) $(TARGET_MCTS_EVAL_GUARDED_MAZE) $(TARGET_MCTS_RUN) $(TARGET_MCTS_RUN_SAILING) $(TARGET_MCTS_RUN_TAXI) $(TARGET_MCTS_RUN_BETTINGGAME) $(TARGET_MCTS_RUN_RISKY_SHORTCUT_GRIDWORLD) $(TARGET_MCTS_RUN_AUTONOMOUS_VEHICLE) $(TARGET_MCTS_RUN_GUARDED_MAZE) $(TARGET_MCTS_RUN_THIN_ICE_FROZEN_LAKE_PLUS) $(TARGET_MCTS_RUN_OVERFLOW_QUEUE) $(TARGET_MCTS_RUN_GAMBLER_JACKPOT) $(TARGET_MCTS_RUN_RISKY_LADDER) $(TARGET_MCTS_RUN_TWO_PATH_SSP_DECEPTIVE) $(TARGET_MCTS_RUN_LASER_TAG_SAFE_GRID) $(TARGET_MCTS_RUN_RIVER_SWIM_STOCHASTIC) $(TARGET_MCTS_RUN_TWO_LEVEL_RISKY_TREASURE) $(TARGET_MCTS_RUN_STREE) $(TARGET_MCTS_VarDE_DBG) $(TARGET_MCTS_DENTS_DBG) $(TARGET_MCTS_VarDE_DBG_SAILING) $(TARGET_MCTS_DENTS_DBG_SAILING) $(TARGET_MCTS_VarDE_DBG_TAXI) $(TARGET_MCTS_DENTS_DBG_TAXI) $(TARGET_MCTS_VarDE_DBG_STREE) $(TARGET_MCTS_DENTS_DBG_STREE)
