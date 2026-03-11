import pytest
import polars as pl
from parallel_rdkit import msready_inchi_inchikey_parallel

def get_msready_inchikeys(smiles_series: pl.Series) -> pl.Series:
    smiles_list = smiles_series.to_list()
    # returns (smiles, inchi, inchikey)
    _, _, inchikeys = msready_inchi_inchikey_parallel(smiles_list)
    return pl.Series(inchikeys)

def test_msready_pubchem_join(msready_csv_path, pubchem_parquet_path):
    if not msready_csv_path or not pubchem_parquet_path:
        pytest.skip("--msready-csv and --pubchem-parquet args required for full test")

    schema = {
        "DTXSID": pl.String,
        "PREFERRED_NAME": pl.String,
        "CASRN": pl.String,
        "DTXCID": pl.String,
        "INCHIKEY": pl.String,
        "IUPAC_NAME": pl.String,
        "SMILES": pl.String,
        "MOLECULAR_FORMULA": pl.String,
        "AVERAGE_MASS": pl.Float64,
        "MONOISOTOPIC_MASS": pl.Float64,
        "QSAR_READY_SMILES": pl.String,
        "MS_READY_SMILES": pl.String,
        "IDENTIFIER": pl.String
    }

    df_csv = (
        pl.scan_csv(msready_csv_path, schema=schema)
        .with_columns(
            pl.col("INCHIKEY").str.split("-").list.get(0).alias("base_inchikey")
        )
    )

    df_pubchem = (
        pl.scan_parquet(str(pubchem_parquet_path), low_memory=True)
        .select(
            [
                pl.col("InChIKey").alias("inchikey_pubchem"),
                pl.col("SMILES").alias("smiles_pubchem"),
            ]
        )
        .with_columns(
            pl.col("inchikey_pubchem").str.split("-").list.get(0).alias("base_inchikey")
        )
        # Deduplicate pubchem by base_inchikey
        .unique(subset="base_inchikey")
    )

    # join them
    mismatches_lf = (
        df_csv.join(df_pubchem, on="base_inchikey")
        .with_columns(
            pl.col("smiles_pubchem")
            .map_batches(get_msready_inchikeys, return_dtype=pl.String)
            .alias("inchikey_msready")
        )
        .with_columns(
            pl.col("inchikey_msready").str.split("-").list.get(0).alias("base_inchikey_msready")
        )
        .filter(pl.col("base_inchikey") != pl.col("base_inchikey_msready"))
        .select(["base_inchikey", "smiles_pubchem", "inchikey_msready", "base_inchikey_msready"])
    )

    # run it all with lazyframes, streaming engine
    mismatches = mismatches_lf.collect(streaming=True)
    
    assert len(mismatches) == 0, f"Found {len(mismatches)} mismatches in base inchikey"

