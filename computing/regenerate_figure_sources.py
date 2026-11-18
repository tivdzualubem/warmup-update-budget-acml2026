"""Regenerate the publication PNGs from the supplied scientific CSVs.

Runs no research training. This also works in the clean scholarly source export.
"""
from pathlib import Path
import runpy,shutil
ROOT=Path(__file__).resolve().parent
for script in ('make_figures.py','make_submission_figures.py','make_peak_figures.py','make_computing_figures.py'):
    runpy.run_path(str(ROOT/script),run_name='__main__')
aliases={
 'fig4_scheduler_and_zero_boundary.png':'fig2_actual_configuration_schedule.png',
 'fig2_selected_paired_endpoints.png':'fig3_selected_paired_endpoints.png',
 'fig3_shared_step_stable_validation.png':'fig4_shared_step_stable_validation.png',
 'fig4_shared_step_legacy_validation.png':'fig5_shared_step_legacy_validation.png',
 'fig8_condition_training_loss.png':'figS2_archived_online_training_loss.png',
 'fig6_ineffective_postzero_steps.png':'figS3_positive_and_zero_lr_budgets.png',
 'fig5_historical_initial_lr_sweep.png':'figS4_historical_initial_lr_severity.png',
 'fig7_guard_threshold_sensitivity.png':'figS5_retrospective_threshold_sensitivity.png',
 'fig1_isolated_first_step_paired.png':'figS6_stable_bad_selected_endpoints.png',
 'fig2_legacy_schedule_order_paired.png':'figS7_legacy_selected_endpoints.png',
 'fig9_exploratory_repair_delay.png':'figS8_exploratory_legacy_endpoints.png',
}
for src,dst in aliases.items():shutil.copy2(ROOT/'figures'/src,ROOT/'figures'/dst)
