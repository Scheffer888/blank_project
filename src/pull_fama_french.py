""" """

import warnings
from pathlib import Path

import pandas as pd
import pandas_datareader.data as web
import wrds

from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))
DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


# ==============================================================================================
# WRDS FAMA FRENCH DATA
# ==============================================================================================


def pull_Fama_French_factors(wrds_username=WRDS_USERNAME, freq="D"):
    conn = wrds.Connection(wrds_username=wrds_username)
    if freq == "D":
        ff = conn.get_table(library="ff", table="factors_daily")
    elif freq == "M":
        ff = conn.get_table(library="ff", table="factors_monthly")
    conn.close()
    return ff


def load_Fama_French_factors(data_dir=DATA_DIR, freq="D"):
    if freq == "D":
        path = Path(data_dir) / "FF_FACTORS_D.parquet"
    elif freq == "M":
        path = Path(data_dir) / "FF_FACTORS_M.parquet"
    ff = pd.read_parquet(path)
    return ff


# ==============================================================================================
# OFFICIAL KEN FRENCH DATABASE
# ==============================================================================================

"""
# Ken French's data library is available through the `pandas_datareader` library.
https://pandas-datareader.readthedocs.io/_/downloads/en/stable/pdf/

# The official website for Ken French's data library is:
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

"""


def pull_ken_french_excel(
    dataset_name="Portfolios_Formed_on_INV",
    data_dir=DATA_DIR,
    log=True,
    start_date=START_DATE,
    end_date=END_DATE,
):
    data_dir = Path(data_dir)
    # Suppress the specific FutureWarning about date_parser
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="The argument 'date_parser' is deprecated",
        )
        data = web.DataReader(
            dataset_name,
            "famafrench",
            start=start_date,
            end=end_date,
        )
        excel_path = (
            data_dir / f"{dataset_name.replace('/', '_')}.xlsx"
        )  # Ensure the name is file-path friendly

        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            # Write the description to the first sheet
            if "DESCR" in data:
                description_df = pd.DataFrame([data["DESCR"]], columns=["Description"])
                description_df.to_excel(writer, sheet_name="Description", index=False)

            # Write each table in the data to subsequent sheets
            for table_key, df in data.items():
                if table_key == "DESCR":
                    continue  # Skip the description since it's already handled
                sheet_name = str(table_key)  # Naming sheets by their table_key
                df.to_excel(
                    writer, sheet_name=sheet_name[:31]
                )  # Sheet name limited to 31 characters
    if log:
        print(f"Excel file saved to {excel_path}")
    return excel_path


def load_ken_french_returns(dataset_name, weighting="value-weighted", data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    excel_path = data_dir / f"{dataset_name.replace('/', '_')}.xlsx"
    if weighting == "value-weighted":
        sheet_name = "0"
    elif weighting == "equal-weighted":
        sheet_name = "1"
    else:
        raise ValueError(f"Invalid weighting: {weighting}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    return df


def load_ken_french_sheet(dataset_name, sheet_name: str = "0", data_dir=DATA_DIR):
    """For example, for dataset_name = 'Portfolios_Formed_on_INV', the full data
    is the description and the 7 tables of returns and properties.

    Portfolios Formed on INV
    ------------------------

    This file was created by CMPT_EP_CFP_OP_INV_NI_AC_RETS using the 202412 CRSP
    database. It contains value- and equal-weighted returns for portfolios
    formed on INV. The portfolios are constructed at the end of June. INV,
    investment, is the change in total assets from the fiscal year ending in
    year t-2 to the fiscal year ending in t-1, divided by t-2 total assets.
    Annual returns are from January to December. Missing data are indicated by
    -99.99 or -999. The break points include utilities and include financials.
    The portfolios include utilities and include financials. Copyright 2024
    Eugene F. Fama and Kenneth R. French

    0 : Value Weight Returns -- Monthly (696 rows x 18 cols)
    1 : Equal Weight Returns -- Monthly (696 rows x 18 cols)
    2 : Value Weight Returns -- Annual from January to December (58 rows x 18 cols)
    3 : Equal Weight Returns -- Annual from January to December (58 rows x 18 cols)
    4 : Number of Firms in Portfolios (696 rows x 18 cols)
    5 : Average Firm Size (696 rows x 18 cols)
    6 : Value Weight Average of Natural Log of INV (58 rows x 18 cols)
    """
    if isinstance(sheet_name, int):
        sheet_name = str(sheet_name)

    data_dir = Path(data_dir)
    excel_path = data_dir / f"{dataset_name.replace('/', '_')}.xlsx"
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if sheet_name == "Description":
        return df.iloc[0, 0]
    return df


def _demo():
    df = load_ken_french_sheet("Portfolios_Formed_on_INV", sheet_name="0")
    df
    df_desc = load_ken_french_sheet("25_Portfolios_OP_INV_5x5", sheet_name="Description")
    print(df_desc)

    ff_factors = load_ken_french_sheet("F-F_Research_Data_Factors", sheet_name="0")
    ff_factors
    ff_factors_desc = load_ken_french_sheet("F-F_Research_Data_Factors", sheet_name="Description")
    print(ff_factors_desc)

    ff_portfolios = load_ken_french_sheet("6_Portfolios_2x3", sheet_name="0")
    ff_portfolios
    ff_portfolios_desc = load_ken_french_sheet("6_Portfolios_2x3", sheet_name="Description")
    print(ff_portfolios_desc)

    ff_dly = load_Fama_French_factors(data_dir=DATA_DIR, freq="D")
    ff_mth = load_Fama_French_factors(data_dir=DATA_DIR, freq="M")


if __name__ == "__main__":
    _ = pull_ken_french_excel(
        dataset_name="Portfolios_Formed_on_INV",
    )  # Save Portfolios_Formed_on_INV.xlsx
    _ = pull_ken_french_excel(
        dataset_name="25_Portfolios_OP_INV_5x5",
    )  # Save 25_Portfolios_OP_INV_5x5.xlsx
    _ = pull_ken_french_excel(
        dataset_name="F-F_Research_Data_Factors",
    )  # Save F-F_Research_Data_Factors.xlsx
    _ = pull_ken_french_excel(
        dataset_name="6_Portfolios_2x3",
    )  # Save 6_Portfolios_2x3.xlsx

    ff_dly = pull_Fama_French_factors(wrds_username=WRDS_USERNAME, freq="D")
    ff_dly.to_parquet(DATA_DIR / "FF_FACTORS_D.parquet")

    ff_mth = pull_Fama_French_factors(wrds_username=WRDS_USERNAME, freq="M")
    ff_mth.to_parquet(DATA_DIR / "FF_FACTORS_M.parquet")