import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--msready-csv", action="store", default=None, help="Path to MSREADY CSV file"
    )
    parser.addoption(
        "--pubchem-parquet", action="store", default=None, help="Path to PubChem Parquet file"
    )

@pytest.fixture
def msready_csv_path(request):
    return request.config.getoption("--msready-csv")

@pytest.fixture
def pubchem_parquet_path(request):
    return request.config.getoption("--pubchem-parquet")
