include ../support/Makefile.inc
include ../support/autoscheduler.inc

BIN ?= bin

all: $(BIN)/process

$(BIN)/depthwise_conv2d_nhwc.generator: depthwise_conv2d_nhwc_generator.cpp $(GENERATOR_DEPS)
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -g $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HALIDE_SYSTEM_LIBS) $(USE_EXPORT_DYNAMIC)

$(BIN)/depthwise_conv2d_nhwc.a: $(BIN)/depthwise_conv2d_nhwc.generator
	@-mkdir -p $(BIN)
	$^ -g depthwise_conv2d_nhwc -o $(BIN) -f depthwise_conv2d_nhwc target=$(HL_TARGET) auto_schedule=false

$(BIN)/depthwise_conv2d_nhwc_classic_auto_schedule.a: $(BIN)/depthwise_conv2d_nhwc.generator $(AUTOSCHED_BIN)/libauto_schedule.so
	@mkdir -p $(@D)
	$< -g depthwise_conv2d_nhwc -f depthwise_conv2d_nhwc_classic_auto_schedule -o $(BIN) target=$(HL_TARGET) auto_schedule=true machine_params=$(HL_MACHINE_PARAMS)

$(BIN)/depthwise_conv2d_nhwc_auto_schedule.a: $(BIN)/depthwise_conv2d_nhwc.generator $(AUTOSCHED_BIN)/libauto_schedule.so
	@mkdir -p $(@D)
	HL_PERMIT_FAILED_UNROLL=1 HL_WEIGHTS_DIR=$(AUTOSCHED_SRC)/weights \
	$< -g depthwise_conv2d_nhwc -f depthwise_conv2d_nhwc_auto_schedule -o $(BIN) target=$(HL_TARGET) auto_schedule=true -p $(AUTOSCHED_BIN)/libauto_schedule.so

$(BIN)/process: process.cpp $(BIN)/depthwise_conv2d_nhwc.a $(BIN)/depthwise_conv2d_nhwc_auto_schedule.a $(BIN)/depthwise_conv2d_nhwc_classic_auto_schedule.a
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -I$(BIN) -Wall -O3 $^ -o $@ $(LDFLAGS)

run: $(BIN)/process
	@-mkdir -p $(BIN)
	$(BIN)/process

clean:
	rm -rf $(BIN)

test: run

autotune: $(BIN)/depthwise_conv2d_nhwc.generator $(AUTOSCHED_BIN)/augment_sample $(AUTOSCHED_BIN)/train_cost_model $(AUTOSCHED_BIN)/libauto_schedule.so $(AUTOSCHED_SRC)/autotune_loop.sh
	bash $(AUTOSCHED_SRC)/autotune_loop.sh \
		$(BIN)/depthwise_conv2d_nhwc.generator \
		depthwise_conv2d_nhwc \
		x86-64-avx2 \
		$(AUTOSCHED_SRC)/weights \
		$(AUTOSCHED_BIN)
