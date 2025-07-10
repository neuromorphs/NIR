# Contributing

NIR is a community-led initiative, and we welcome contributions from everyone.
Here, we outline some technical details on getting started.
Join the conversation on our [Discord server](https://discord.gg/JRMRGP9h3c) or [GitHub](https://github.com/neuromorphs/nir) if you have any questions.

## Developer guide: Getting started

Use the standard github workflow.

1. Fork the repository.

2. Setup the virtual environment for this project.

3. Install all the development requirements (refer to the options described below).

4. Install git pre-commit hooks.

```shell
pre-commit install 
```

5. Now you are all set. Go ahead, make your changes, test the features using `pytest` and commit them.

6. [Create a pull request from your fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)

## Installing the development requirements

### Using uv

Easy package management can be accomplished using the [uv](https://docs.astral.sh/uv/) tool which will need to be available on
your local workstation. As explained in [the official uv documentation](https://docs.astral.sh/uv/getting-started/installation/),
a number of straightforward installation options are available. uv is supported by IDEs such as [PyCharm](https://www.jetbrains.com/help/pycharm/uv.html).


After the first clone of this repository, run the following command in the root directory. 
This will automatically create
a new virtual environment in the folder `.venv` and install all required NIR project dependencies to it.  

```shell
uv sync
```


To run commands within the uv virtual environment, prefix them with `uv run`. 
For example, to run all the tests:

```shell
uv run pytest
```

Alternatively, you can activate the virtual environment in your current terminal and run commands directly.
For example:
```shell
source .venv/bin/activate
pytest
```

### Using pip 

```shell
pip install -r dev_requirements.txt
```

## Code formatting

We use `black` to format the code and `ruff` to linting.
The rules and formatting are embedded in the `pre-commit hooks`. So you do not need to explicitly worry about these but is good to know when you see erros while commiting your code or in the CI.