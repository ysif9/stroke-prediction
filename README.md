# Stroke Prediction Classification Model

A machine learning project to predict stroke likelihood in patients based on clinical and demographic parameters. This
project applies data preprocessing, outlier detection, classification algorithms, and comprehensive data visualization
to build and evaluate predictive models.

## Setup & Installation

### Installing uv

`uv` is a fast Python package installer and resolver. Install it following
the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/ysif9/stroke-prediction.git
   cd stroke-prediction
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Start Jupyter Notebook**
    * Option 1: Run notebook in IDE
    * Option 2: Run notebook in browser
      ```bash
      uv run --with jupyter jupyter lab
      ```

### Using uv for Project Management

| Task                       | pip                                   | uv                          |
|----------------------------|---------------------------------------|-----------------------------|
| Install dependencies       | `pip install -r requirements.txt`     | `uv sync`                   |
| Add a package              | `pip install package_name`            | `uv add package_name`       |
| Add dev dependency         | `pip install --save-dev package_name` | `uv add --dev package_name` |
| Freeze dependencies        | `pip freeze > requirements.txt`       | `uv lock`                   |
| Run a script               | `python script.py`                    | `uv run script.py`          |

