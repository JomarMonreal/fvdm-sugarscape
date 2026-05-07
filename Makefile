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
		$(TESTS)

# Change to python3 (or other alias) if needed
PYTHON = python3
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

clean:
	rm -rf $(CLEAN) || true

lean:
	rm -rf $(PLOTS) || true

# ─────────────────────────────────────────────────────────────────
# Experiment runner (run_experiments.py)
#
# Configurable flags (override on command line, e.g. make experiment SEEDS=10):
#   CORES     — parallel CPU cores            (default: 1)
#   SEEDS     — seeds per condition           (default: 500)
#   AGENTS    — starting agents per run       (default: 250)
#   TIMESTEPS — simulation timesteps per run  (default: 5000)
#   GUI       — if set to 1, launch GUI run   (default: 0)
#   EXP_OUT   — output directory              (default: experiment_results)
# ─────────────────────────────────────────────────────────────────

CORES     ?= 1
SEEDS     ?= 500
AGENTS    ?= 250
TIMESTEPS ?= 5000
GUI       ?= 0
EXP_OUT   ?= experiment_results
EXP_RUNNER = run_experiments.py

ifeq ($(GUI), 1)
GUI_FLAG = --gui
else
GUI_FLAG =
endif

experiment:
	$(PYTHON) $(EXP_RUNNER) \
		--config $(CONFIG) \
		--output $(EXP_OUT) \
		--seeds $(SEEDS) \
		--agents $(AGENTS) \
		--timesteps $(TIMESTEPS) \
		--cores $(CORES) \
		--python $(PYTHON) \
		$(GUI_FLAG)

experiment-gui:
	$(PYTHON) $(EXP_RUNNER) \
		--config $(CONFIG) \
		--output $(EXP_OUT) \
		--seeds 1 \
		--agents $(AGENTS) \
		--timesteps $(TIMESTEPS) \
		--cores 1 \
		--python $(PYTHON) \
		--gui

experiment-clean:
	rm -rf $(EXP_OUT)

visualize:
	$(PYTHON) visualize_results.py \
		--results $(EXP_OUT)/results \
		--output $(EXP_OUT)/results/figures

sanity-test:
	$(MAKE) experiment SEEDS=10 TIMESTEPS=200 EXP_OUT=experiment_results_test
	$(MAKE) visualize EXP_OUT=experiment_results_test

.PHONY: all clean data experiment experiment-clean experiment-gui lean plots run seeds setup test visualize sanity-test
# vim: set noexpandtab tabstop=4:
