import mlflow

# Check for an active run
active_run = mlflow.active_run()
if active_run:
    print(f"Ending active run: {active_run.info.run_id}")
    mlflow.end_run()