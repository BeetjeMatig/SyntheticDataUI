# SyntheticDataUI

SyntheticDataUI is a Django-based web application for generating synthetic data.

## Features
- Upload a CSV file and edit the data directly in the browser.
- Generate synthetic records using the algorithms defined in `synth_ui/frontend/synth.py`.
- Download the generated synthetic data or evaluation metrics.

## Setup

### Create the Conda environment
1. Install [Conda](https://docs.conda.io/) if you don't already have it.
2. Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
```

This will create an environment named **SYNTHETIC** with all required dependencies.
Activate it with:

```bash
conda activate SYNTHETIC
```

### Database migrations
Run the initial migrations to set up the SQLite database:

```bash
python synth_ui/manage.py migrate
```

### Launch the development server
Start the Django development server:

```bash
python synth_ui/manage.py runserver
```

Then open your browser to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to upload a CSV and try generating synthetic records.

## Running tests

Execute the Django test suite with:

```bash
python synth_ui/manage.py test
```

## Project structure
- `synth_ui/` – Django project and application code.
- `environment.yml` – Conda environment with all dependencies for synthetic data generation.

