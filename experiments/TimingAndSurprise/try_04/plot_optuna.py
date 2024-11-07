import optuna
from optuna import create_study
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_slice, plot_contour
from optuna.importance import FanovaImportanceEvaluator
import matplotlib.pyplot as plt

# Load the original study
original_study_name = "no-name-f1b1926e-7319-45ff-9b24-d31b65de35e1"
original_study = optuna.load_study(study_name=original_study_name, storage="sqlite:///optuna_study.db")

# Create a temporary study with capped objective values
temp_study = create_study(direction="minimize")
for trial in original_study.trials:
    if trial.state == TrialState.COMPLETE:
        if trial.value is not None and trial.value > 2:
            continue # Skip trials with objective values greater than 2
        #else
        #capped_value = min(trial.value, 2) if trial.value is not None else None
        temp_study.add_trial(
            optuna.trial.create_trial(
                params=trial.params,
                distributions=trial.distributions,
                value=trial.value,
                state=trial.state
            )
        )

# Plot optimization history with capped values
fig1 = plot_optimization_history(temp_study)
fig1.update_layout(title="Optimization History with Objective Values Capped at 4")
fig1.show()

# Instantiate FANOVA evaluator
fanova_evaluator = FanovaImportanceEvaluator()

# Calculate and display parameter importances using FANOVA
param_importances = fanova_evaluator.evaluate(temp_study)
print("Parameter importances with FANOVA:", param_importances)

# Plot parameter importances using FANOVA
fig2 = plot_param_importances(temp_study, evaluator=fanova_evaluator)
fig2.update_layout(title="Parameter Importances with Capped Objective Values (FANOVA)")
fig2.show()

# Parallel coordinate plot
fig3 = plot_parallel_coordinate(temp_study)
fig3.update_layout(title="Parallel Coordinate Plot with Capped Objective Values")
fig3.show()

# Slice plot
fig4 = plot_slice(temp_study)
fig4.update_layout(title="Slice Plot with Capped Objective Values")
fig4.show()

# Contour plot
fig5 = plot_contour(temp_study)
fig5.update_layout(title="Contour Plot with Capped Objective Values")
fig5.show()
