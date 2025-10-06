# Building dist packages and run tests

```shell
set -o errexit
python -m virtualenv venv
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade build
pip install --upgrade virtualenv
if [ -f dev_requirements.txt ]; then pip install -r dev_requirements.txt; fi
python3 -m build
pip install dist/*.whl
# Lint with Ruff
ruff check . --select=E9,F63,F7,F82 --target-version=py37
ruff check . --target-version=py37 --exclude=docs/ --exclude=paper/
# Check formatting with Black
. venv/bin/activate
black --check nir/ tests/
# Run Tests
. venv/bin/activate
pytest
```
