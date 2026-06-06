# TODO - CI/Docker & MLflow Project Fix

## Step 1: Update modelling.py (path robustness)
- [ ] Make data path resolve from repository root using `Path(__file__).resolve()`.
- [ ] Add clear error message if expected columns/files are missing.
- [ ] Ensure works under both local run and MLflow docker context.

## Step 2: Update .github/workflows/train_ci.yml
- [ ] Use `--env-manager conda` instead of `local`.
- [ ] Replace heredoc docker build with Dockerfile-based build.

## Step 3: Add MLProject_Folder/Dockerfile
- [ ] Create Dockerfile that installs deps and runs `python modelling.py`.
- [ ] Set WORKDIR so modelling.py sees repo structure.

## Step 4: Smoke test locally
- [ ] Run `python MLProject_Folder/modelling.py` (without DAGSHUB_TOKEN).
- [ ] Optionally build Docker image locally and run container.

## Step 5: Validate CI logic
- [ ] Verify `mlflow run MLProject_Folder ...` works with conda env.
- [ ] Ensure workflow YAML syntax is valid.

