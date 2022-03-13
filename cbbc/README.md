# Capability Boundary Breakthrough Curriculum (CBBC) pipeline

## Prerequisite
```bash
pip install -r requirements.txt
```

## Evaluate a sample with the 4-dimensional capability-specific value map

- Follow the [README]() in the `/competency_metrics/` sub-directory.

## Assess a model's competency from its 4-dimensional capability and boost its capability gradually

- Configure fromat
    - We follow the widely-used hierarchical configuration paradigm using the [YAML](https://en.wikipedia.org/wiki/YAML) (a human-readable data-serialization mark language) file format.
    - Run the following script to create a default configure file.
    ```bash
    cd configure/
    python default.py
    ```
    - Adapt the specific items (or entries) according to your requirements. We have provided four typical configurations in the configure directory.

- Train then test
```bash
python source/main.py --cfg path/to/configure
```

- Test (including inference and evaluation)
```bash
python source/predict.py --cfg path/to/configure
```