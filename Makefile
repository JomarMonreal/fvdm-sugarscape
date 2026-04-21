CONFIG = config.json
DATACHECK = data/data.complete
LOGS = agents.log.csv agents.log.json log.csv log.json
PLOT = plot.py
PLOTCHECK = plots/plots.complete
RUN = run.py
SCREENSHOTS = *.ps
TEST = test.py

DATASET = $(DATACHECK) \
		data/*[[:digit:]]*.config \
		data/*.csv \
		data/*.json \
		data/*.sh

PLOTS = $(PLOTCHECK) \
		plots/*.pdf

TESTS = tests/*.config \
        tests/*.log

CLEAN = $(DATASET) \
		$(LOGS) \
		$(PLOTS) \
		$(SCREENSHOTS) \
		$(TESTS) \
		observations/ \
		results/fvdm_weights.json

# Change to python3 (or other alias) if needed
PYTHON = ./.venv/bin/python3
SUGARSCAPE = sugarscape.py

# Check for local Python aliases
PYCHECK = $(shell which python > /dev/null; echo $$?)
PY3CHECK = $(shell which python3 > /dev/null; echo $$?)

$(DATACHECK):
	cd data && $(PYTHON) $(RUN) --conf ../$(CONFIG) --mode csv
	touch $(DATACHECK)

$(PLOTCHECK): $(DATACHECK)
	cd plots && $(PYTHON) $(PLOT) --path ../data/ --conf ../$(CONFIG)
	touch $(PLOTCHECK)

all: $(DATACHECK) $(PLOTCHECK)

data: $(DATACHECK)

plots: $(PLOTCHECK)

run:
	$(PYTHON) $(SUGARSCAPE) --conf $(CONFIG)

seeds:
	cd data && $(PYTHON) $(RUN) --conf ../$(CONFIG) --mode csv --seeds

setup:
	@echo "Checking for local Python installation."
ifeq ($(PY3CHECK), 0)
	@echo "Found alias for Python."
	sed -i 's/PYTHON = python$$/PYTHON = python3/g' Makefile
	sed -i 's/"python"/"python3"/g' $(CONFIG)
else ifneq ($(PYCHECK), 0)
	@echo "Could not find a local Python installation."
	@echo "Please update the Makefile and configuration file manually."
else
	@echo "This message should never be reached."
endif

test:
	cd tests && $(PYTHON) $(TEST) --conf ../$(CONFIG)

SEEDS_DEMO ?= 10
SEEDS_MAIN ?= 500
CORES ?= $(shell nproc)
BATCHES ?= 1
AGENTS ?= 250
EXPERIMENT_SEEDS ?= 500

demo-horizon:
	$(PYTHON) horizon_calibration.py --demo --seeds $(SEEDS_DEMO) --processes $(CORES) --agents $(AGENTS)

main-horizon:
	$(PYTHON) horizon_calibration.py --baseline --seeds $(SEEDS_MAIN) --processes $(CORES) --agents $(AGENTS)

demo-derive:
	$(PYTHON) felicific_derivation.py --demo --seeds 5 --processes $(CORES) --agents $(AGENTS)

demo-collect:
	$(PYTHON) felicific_derivation.py --collect --demo --seeds 5 --processes $(CORES) --agents $(AGENTS)

derive-felicific:
	$(PYTHON) felicific_derivation.py --seeds $(SEEDS_MAIN) --processes $(CORES) --agents $(AGENTS)

derive-collect:
	$(PYTHON) felicific_derivation.py --collect --seeds $(SEEDS_MAIN) --processes $(CORES) --agents $(AGENTS)

derive-train:
	$(PYTHON) felicific_derivation.py --train --batches $(BATCHES)

derive-clean:
	rm -rf observations/

demo-prioritize:
	$(PYTHON) prioritization_derivation.py --demo --seeds 10 --processes $(CORES) --agents $(AGENTS)

derive-prioritization:
	$(PYTHON) prioritization_derivation.py --seeds $(SEEDS_MAIN) --processes $(CORES) --agents $(AGENTS)

FVDM_CONF ?= configs/demo_fvdm.json

## Run individual FVDM ethical agents
demo-fvdm-selfish:
	$(PYTHON) $(SUGARSCAPE) --gui --conf configs/demo_fvdm_selfish.json

demo-fvdm-altruist:
	$(PYTHON) $(SUGARSCAPE) --gui --conf configs/demo_fvdm_altruist.json

demo-fvdm-bentham:
	$(PYTHON) $(SUGARSCAPE) --gui --conf configs/demo_fvdm_bentham.json

## Run all three FVDM models together in one mixed simulation
demo-fvdm:
	$(PYTHON) $(SUGARSCAPE) --gui --conf $(FVDM_CONF)

## Stage 5: Comparison Runs
test-homogeneous:
	$(PYTHON) $(SUGARSCAPE) --headless --conf configs/test_homo_$(ENV)_$(MODEL).json

test-heterogeneous:
	$(PYTHON) $(SUGARSCAPE) --headless --conf configs/test_hetero_$(ENV).json

run-experiments:
	$(PYTHON) run_experiments.py --seeds $(EXPERIMENT_SEEDS) --processes $(CORES)

evaluate-experiments:
	$(PYTHON) aggregate_evaluations.py results/experiments/

.PHONY: all clean data lean plots run seeds setup test demo-horizon main-horizon demo-derive demo-collect derive-felicific derive-collect derive-train derive-clean demo-prioritize derive-prioritization demo-fvdm demo-fvdm-selfish demo-fvdm-altruist demo-fvdm-bentham test-homogeneous test-heterogeneous run-experiments evaluate-experiments
# vim: set noexpandtab tabstop=4:
