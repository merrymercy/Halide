include ../support/Makefile.inc
include ../support/autoscheduler.inc

BIN ?= bin

all: $(BIN)/process

$(BIN)/conv2d_capsule_nhwijc.generator: conv2d_capsule_nhwijc_generator.cpp $(GENERATOR_DEPS)
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -g $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HALIDE_SYSTEM_LIBS) $(USE_EXPORT_DYNAMIC)

$(BIN)/conv2d_capsule_nhwijc.a: $(BIN)/conv2d_capsule_nhwijc.generator
	@-mkdir -p $(BIN)
	$^ -g conv2d_capsule_nhwijc -o $(BIN) -f conv2d_capsule_nhwijc target=$(HL_TARGET) auto_schedule=false

$(BIN)/conv2d_capsule_nhwijc_classic_auto_schedule.a: $(BIN)/conv2d_capsule_nhwijc.generator $(AUTOSCHED_BIN)/libauto_schedule.so
	@mkdir -p $(@D)
	$< -g conv2d_capsule_nhwijc -f conv2d_capsule_nhwijc_classic_auto_schedule -o $(BIN) target=$(HL_TARGET) auto_schedule=true machine_params=$(HL_MACHINE_PARAMS)

$(BIN)/conv2d_capsule_nhwijc_auto_schedule.a: $(BIN)/conv2d_capsule_nhwijc.generator $(AUTOSCHED_BIN)/libauto_schedule.so
	@mkdir -p $(@D)
	HL_PERMIT_FAILED_UNROLL=1 HL_WEIGHTS_DIR=$(AUTOSCHED_SRC)/weights \
	$< -g conv2d_capsule_nhwijc -f conv2d_capsule_nhwijc_auto_schedule -o $(BIN) target=$(HL_TARGET) auto_schedule=true -p $(AUTOSCHED_BIN)/libauto_schedule.so

$(BIN)/process: process.cpp $(BIN)/conv2d_capsule_nhwijc.a $(BIN)/conv2d_capsule_nhwijc_auto_schedule.a $(BIN)/conv2d_capsule_nhwijc_classic_auto_schedule.a
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -I$(BIN) -Wall -O3 $^ -o $@ $(LDFLAGS)

run: $(BIN)/process
	@-mkdir -p $(BIN)
	$(BIN)/process

clean:
	rm -rf $(BIN)

test: run

autotune: $(BIN)/conv2d_capsule_nhwijc.generator $(AUTOSCHED_BIN)/augment_sample $(AUTOSCHED_BIN)/train_cost_model $(AUTOSCHED_BIN)/libauto_schedule.so $(AUTOSCHED_SRC)/autotune_loop.sh
	bash $(AUTOSCHED_SRC)/autotune_loop.sh \
		$(BIN)/conv2d_capsule_nhwijc.generator \
		conv2d_capsule_nhwijc \
		x86-64-avx2 \
		$(AUTOSCHED_SRC)/weights \
		$(AUTOSCHED_BIN)
