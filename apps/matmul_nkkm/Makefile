include ../support/Makefile.inc

.PHONY: build clean test

build: $(BIN)/$(HL_TARGET)/process

$(GENERATOR_BIN)/matmul.generator: matmul_generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LIBHALIDE_LDFLAGS)

$(BIN)/%/matmul.a: $(GENERATOR_BIN)/matmul.generator
	@mkdir -p $(@D)
	$^ -g matmul -e $(GENERATOR_OUTPUTS) -o $(@D) -f matmul target=$* auto_schedule=false

$(BIN)/%/matmul_auto_schedule.a: $(GENERATOR_BIN)/matmul.generator
	@mkdir -p $(@D)
	$^ -g matmul -e $(GENERATOR_OUTPUTS) -o $(@D) -f matmul_auto_schedule target=$*-no_runtime auto_schedule=true

$(BIN)/%/process: process.cpp $(BIN)/%/matmul.a $(BIN)/%/matmul_auto_schedule.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -Wall $^ -o $@ $(LDFLAGS)

run: $(BIN)/$(HL_TARGET)/process
	@mkdir -p $(@D)
	$^

clean:
	rm -rf $(BIN)

test: run
