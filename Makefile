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
FVDM_RUNNER = run_fvdm.py
HETERO_RUNNER = run_hetero.py
DATA_ROOT = data
SCREENSHOTS = *.ps
TEST = test.py

# Experiment options (override on command line, e.g. SEEDS=10 TIMESTEPS=500)
SEEDS      = 30
TIMESTEPS  = 1000
AGENTS     = 250
PARALLEL   = $(shell nproc 2>/dev/null || echo 1)
# Derivation-phase options
DERIVE_TIMESTEPS = 5000
DERIVE_SEEDS     = 30
DERIVE_PARALLEL  = 30

DATASET = $(DATACHECK) \
		data/*[[:digit:]]*.config \
		data/*.csv \
		data/*.json \
		data/*.sh

PLOTS = $(PLOTCHECK) \
		plots/*.pdf

TESTS = tests/*.config \
        tests/*.log

FVDM_DIRS = $(DATA_ROOT)/fvdmRawSugarscape \
            $(DATA_ROOT)/fvdmEgoist \
            $(DATA_ROOT)/fvdmAltruist \
            $(DATA_ROOT)/fvdmBentham

HETERO_DIRS = $(DATA_ROOT)/rawSugarscape \
              $(DATA_ROOT)/egoist \
              $(DATA_ROOT)/altruist \
              $(DATA_ROOT)/bentham \
              $(FVDM_DIRS)

CLEAN = $(DATASET) \
		$(LOGS) \
		$(PLOTS) \
		$(SCREENSHOTS) \
		$(TESTS) \
		$(BASELINE_DIR) \
		$(FVDM_DIRS) \
		$(HETERO_DIRS)

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
	$(PYTHON) $(DERIVE) --seeds $(DERIVE_SEEDS) --timesteps $(DERIVE_TIMESTEPS) --parallel $(DERIVE_PARALLEL) --outdir $(BASELINE_DIR)

derive-force:
	$(PYTHON) $(DERIVE) --seeds $(DERIVE_SEEDS) --timesteps $(DERIVE_TIMESTEPS) --parallel $(DERIVE_PARALLEL) --outdir $(BASELINE_DIR) --force

fvdm:
	$(PYTHON) $(FVDM_RUNNER) --seeds $(SEEDS) --timesteps $(TIMESTEPS) --agents $(AGENTS) --parallel $(PARALLEL) --outdir $(DATA_ROOT) --baseline-dir $(BASELINE_DIR)

fvdm-force:
	$(PYTHON) $(FVDM_RUNNER) --seeds $(SEEDS) --timesteps $(TIMESTEPS) --agents $(AGENTS) --parallel $(PARALLEL) --outdir $(DATA_ROOT) --baseline-dir $(BASELINE_DIR) --force

hetero:
	$(PYTHON) $(HETERO_RUNNER) --seeds $(SEEDS) --timesteps $(TIMESTEPS) --agents $(AGENTS) --parallel $(PARALLEL) --outdir $(DATA_ROOT) --baseline-dir $(BASELINE_DIR)

hetero-force:
	$(PYTHON) $(HETERO_RUNNER) --seeds $(SEEDS) --timesteps $(TIMESTEPS) --agents $(AGENTS) --parallel $(PARALLEL) --outdir $(DATA_ROOT) --baseline-dir $(BASELINE_DIR) --force

all_fvdm: baseline derive fvdm

all_fvdm-force:
	$(MAKE) baseline-force
	$(MAKE) derive-force
	$(MAKE) fvdm-force

test:
	cd tests && $(PYTHON) $(TEST) --conf ../$(CONFIG)

clean:
	rm -rf $(CLEAN) || true

lean:
	rm -rf $(PLOTS) || true

.PHONY: all all_fvdm all_fvdm-force baseline baseline-force clean data derive derive-force fvdm fvdm-force hetero hetero-force lean plots run seeds setup test
# vim: set noexpandtab tabstop=4:
