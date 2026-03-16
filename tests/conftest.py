import pytest

from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "-D", "--dump", action="store", default=None,
        help="Path to MSREADY CSV file or directory containing CSV files"
    )
    parser.addoption(
        "-P", "--pubchem-parquet", action="store", default=None,
        help="Path to PubChem Parquet file"
    )
    parser.addoption(
        "-N", "--num-examples", action="store", type=int, default=None,
        help="Number of examples to take from the dump (for testing)"
    )


@pytest.fixture
def dump_path(request):
    return request.config.getoption("--dump")


@pytest.fixture
def msready_csv_path(dump_path):
    if dump_path is None:
        return None
    path = Path(dump_path)
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        return [str(f) for f in csv_files]
    else:
        return [str(path)]


@pytest.fixture
def pubchem_parquet_path(request):
    return request.config.getoption("--pubchem-parquet")


@pytest.fixture
def num_examples(request):
    return request.config.getoption("--num-examples")
