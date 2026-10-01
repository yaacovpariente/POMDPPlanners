# SPDX-License-Identifier: MIT

"""Where a tuning study leaves its record in MLflow.

The optimizer writes these names and the results site reads them. They live
in one small module with no heavy imports, so the site can read them without
importing Optuna or the simulator, and so the two sides cannot drift apart.

A study run by ``optimize_and_evaluate`` leaves three kinds of run in one
experiment:

- the **study** run (``optimize_batch_<n>_configs``), the parent of the rest;
- one **config** run per tuned planner and environment, holding the best
  parameters, the best trial's own scores, and the files under
  :data:`TUNING_ARTIFACT_DIR`;
- the **evaluation** run, every chosen planner run again on the
  evaluation episode count. Each config
  run names it in the tag :data:`EVALUATION_RUN_ID_TAG`.
"""

# Tag on every run a study writes, saying which of the three kinds it is.
RUN_KIND_TAG = "pomdp.run_kind"
RUN_KIND_STUDY = "tuning_study"
RUN_KIND_CONFIG = "tuning_config"
RUN_KIND_EVALUATION = "tuning_evaluation"

# On a config run: the id of the run that evaluated its chosen planner afresh.
EVALUATION_RUN_ID_TAG = "pomdp.tuning.evaluation_run_id"

# On a config run: the folder of tuning files, and the files in it.
TUNING_ARTIFACT_DIR = "tuning"
STUDY_SUMMARY_FILE = "study_summary.json"
TRIAL_RECORDS_FILE = "trial_records.json"

# Metric and param prefixes on a config run.
BEST_PARAM_PREFIX = "best_"
PARAM_RANGE_PREFIX = "param_range_"
BEST_TRIAL_METRIC_PREFIX = "best_trial_"
