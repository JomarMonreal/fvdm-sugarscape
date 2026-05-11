CONFIG = config.json
DATACHECK = data/data.complete
LOGS = agents.log.csv agents.log.json log.csv log.json
PLOT = plot.py
PLOTCHECK = plots/plots.complete
RUN = run.py
BASELINE = run_baseline_conditions.py
BASELINE_DIR = data/baseline
DERIVE = derive_vectors.py
VECTORS = $(BASELINE_DIR)/prioritization_vectors.json
SCREENSHOTS = *.ps
TEST = test.py

# Experiment options (override on command line, e.g. SEEDS=10 TIMESTEPS=500)
SEEDS      = 15
TIMESTEPS  = 1000
AGENTS     = 250
PARALLEL   = $(shell nproc 2>/dev/null || echo 1)
# Derivation-phase options (longer runs for IRL convergence)
DERIVE_TIMESTEPS          = 2000
DERIVE_LR                 = 0.1
DERIVE_CONVERGE_THRESHOLD = 1e-3
DERIVE_CONVERGE_PATIENCE  = 3

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
		$(BASELINE_DIR)

# Change to python3 (or other alias) if needed
PYTHON = python
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

baseline:
	$(PYTHON) $(BASELINE) --seeds $(SEEDS) --timesteps $(TIMESTEPS) --agents $(AGENTS) --parallel $(PARALLEL) --outdir $(BASELINE_DIR)

baseline-force:
	$(PYTHON) $(BASELINE) --seeds $(SEEDS) --timesteps $(TIMESTEPS) --agents $(AGENTS) --parallel $(PARALLEL) --outdir $(BASELINE_DIR) --force

derive:
	$(PYTHON) $(DERIVE) --seeds $(SEEDS) --timesteps $(DERIVE_TIMESTEPS) --lr $(DERIVE_LR) --converge-threshold $(DERIVE_CONVERGE_THRESHOLD) --converge-patience $(DERIVE_CONVERGE_PATIENCE) --parallel $(PARALLEL) --outdir $(BASELINE_DIR)

derive-force:
	$(PYTHON) $(DERIVE) --seeds $(SEEDS) --timesteps $(DERIVE_TIMESTEPS) --lr $(DERIVE_LR) --converge-threshold $(DERIVE_CONVERGE_THRESHOLD) --converge-patience $(DERIVE_CONVERGE_PATIENCE) --parallel $(PARALLEL) --outdir $(BASELINE_DIR) --force

test:
	cd tests && $(PYTHON) $(TEST) --conf ../$(CONFIG)

clean:
	rm -rf $(CLEAN) || true

lean:
	rm -rf $(PLOTS) || true

.PHONY: all baseline baseline-force clean data derive derive-force lean plots run seeds setup test
# vim: set noexpandtab tabstop=4:
