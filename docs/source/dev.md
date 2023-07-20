# Contributing to NIR

## Developer guide: Getting started

Use the standard github workflow.

1. Fork the repository.

2. Setup the virtual environment for this project.

3. Install all the development requirements.

```shell
pip install -r dev_requirements.txt
```

4. Install git pre-commit hooks.

```shell
pre-commit install 
```

5. Now you are all set. Go ahead, make your changes, test the features using `pytest` and commit them.

6. [Create a pull request from your fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)

## Code formatting.

We use `black` to format the code and `ruff` to linting.
The rules and formatting are embedded in the `pre-commit hooks`. So you do not need to explicitly worry about these but is good to know when you see erros while commiting your code or in the CI.