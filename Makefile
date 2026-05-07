CXX ?= g++

SRC_DIR := src
BIN_DIR := bin

DEPFLAGS := -MMD -MP
CPPFLAGS := -Iinclude -Isrc -I. $(DEPFLAGS)
CXXFLAGS ?= -Wall -std=c++20
LDLIBS := -lpthread

TARGET_MCTS := mcts
TARGET_MCTS_RUN := mcts-run
TARGET_MCTS_TUNE := mcts-tune
TARGET_MCTS_EVAL_AUTONOMOUS_VEHICLE := mcts-eval-autonomous-vehicle
TARGET_MCTS_EVAL_RISKY_SHORTCUT_GRIDWORLD := mcts-eval-risky-shortcut-gridworld

GENERIC_TARGETS := \
	$(TARGET_MCTS_RUN) \
	$(TARGET_MCTS_TUNE)

EVAL_TARGETS := \
	$(TARGET_MCTS_EVAL_AUTONOMOUS_VEHICLE) \
	$(TARGET_MCTS_EVAL_RISKY_SHORTCUT_GRIDWORLD)

ALL_TARGETS := \
	$(GENERIC_TARGETS) \
	$(EVAL_TARGETS)

to_objects = \
	$(patsubst $(SRC_DIR)/%.cpp,$(BIN_DIR)/$(SRC_DIR)/%.o,$(filter %.cpp,$1)) \
	$(patsubst $(SRC_DIR)/%.cc,$(BIN_DIR)/$(SRC_DIR)/%.o,$(filter %.cc,$1))

COMMON_CPP_SOURCES := \
	$(wildcard $(SRC_DIR)/*.cpp) \
	$(wildcard $(SRC_DIR)/algorithms/common/*.cpp) \
	$(wildcard $(SRC_DIR)/algorithms/uct/*.cpp) \
	$(wildcard $(SRC_DIR)/algorithms/catso/*.cpp) \
	$(wildcard $(SRC_DIR)/env/*.cpp)

COMMON_SOURCES := $(sort $(COMMON_CPP_SOURCES))
COMMON_OBJECTS := $(call to_objects,$(COMMON_SOURCES))

EXP_COMMON_SOURCES := \
	$(SRC_DIR)/exp/algorithm_factory.cpp \
	$(SRC_DIR)/exp/env_registry.cpp \
	$(SRC_DIR)/exp/evaluation_utils.cpp \
	$(SRC_DIR)/exp/experiment_runner.cpp \
	$(SRC_DIR)/exp/register_all_envs.cpp \
	$(SRC_DIR)/exp/tuning_runner.cpp \
	$(wildcard $(SRC_DIR)/exp/env_specs/*.cpp) \
	$(SRC_DIR)/exp/oracles/autonomous_vehicle_cvar_oracle.cpp

EXP_COMMON_OBJECTS := $(call to_objects,$(EXP_COMMON_SOURCES))

RUN_SOURCES := $(SRC_DIR)/exp/run_experiment.cpp
RUN_OBJECTS := $(call to_objects,$(RUN_SOURCES))

TUNE_SOURCES := $(SRC_DIR)/exp/tune_experiment.cpp
TUNE_OBJECTS := $(call to_objects,$(TUNE_SOURCES))

AUTONOMOUS_VEHICLE_EVAL_SOURCES := $(SRC_DIR)/exp/eval_autonomous_vehicle.cpp
AUTONOMOUS_VEHICLE_EVAL_OBJECTS := $(call to_objects,$(AUTONOMOUS_VEHICLE_EVAL_SOURCES))

RISKY_SHORTCUT_GRIDWORLD_EVAL_SOURCES := $(SRC_DIR)/exp/eval_risky_shortcut_gridworld.cpp
RISKY_SHORTCUT_GRIDWORLD_EVAL_OBJECTS := $(call to_objects,$(RISKY_SHORTCUT_GRIDWORLD_EVAL_SOURCES))

DEPFILES := \
	$(COMMON_OBJECTS:.o=.d) \
	$(EXP_COMMON_OBJECTS:.o=.d) \
	$(RUN_OBJECTS:.o=.d) \
	$(TUNE_OBJECTS:.o=.d) \
	$(AUTONOMOUS_VEHICLE_EVAL_OBJECTS:.o=.d) \
	$(RISKY_SHORTCUT_GRIDWORLD_EVAL_OBJECTS:.o=.d)

all: $(ALL_TARGETS)

generic: $(GENERIC_TARGETS)

eval: $(EVAL_TARGETS)

help:
	@printf '%s\n' \
		'Available targets:' \
		'  all                               Build generic runners and evaluator binaries' \
		'  generic                           Build mcts-run and mcts-tune' \
		'  eval                              Build evaluator binaries' \
		'  $(TARGET_MCTS)                              Build shared object files only' \
		'  $(TARGET_MCTS_RUN)                          Build the generic experiment runner' \
		'  $(TARGET_MCTS_TUNE)                         Build the generic hyperparameter tuner' \
		'  $(TARGET_MCTS_EVAL_AUTONOMOUS_VEHICLE)          Build the Autonomous Vehicle evaluator' \
		'  $(TARGET_MCTS_EVAL_RISKY_SHORTCUT_GRIDWORLD)    Build the Risky Shortcut Gridworld evaluator' \
		'  Supported generic --env values: autonomous_vehicle, risky_shortcut_gridworld, two_level_risky_treasure' \
		'  clean                             Remove build artifacts'

$(TARGET_MCTS): $(COMMON_OBJECTS) $(EXP_COMMON_OBJECTS)

$(TARGET_MCTS_RUN): $(COMMON_OBJECTS) $(EXP_COMMON_OBJECTS) $(RUN_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TARGET_MCTS_TUNE): $(COMMON_OBJECTS) $(EXP_COMMON_OBJECTS) $(TUNE_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TARGET_MCTS_EVAL_AUTONOMOUS_VEHICLE): $(COMMON_OBJECTS) $(AUTONOMOUS_VEHICLE_EVAL_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(TARGET_MCTS_EVAL_RISKY_SHORTCUT_GRIDWORLD): $(COMMON_OBJECTS) $(RISKY_SHORTCUT_GRIDWORLD_EVAL_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(BIN_DIR)/$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(BIN_DIR)/$(SRC_DIR)/%.o: $(SRC_DIR)/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	$(RM) -r $(BIN_DIR)
	$(RM) $(ALL_TARGETS)

-include $(DEPFILES)

.PHONY: all clean eval generic help $(TARGET_MCTS)
