"""
This module pulls and saves data on fundamentals from CRSP and Compustat.
It pulls fundamentals data from Compustat needed to calculate
book equity, and the data needed from CRSP to calculate market equity.

Note: This code uses the new CRSP CIZ format. Information
about the differences between the SIZ and CIZ format can be found here:

 - Transition FAQ: https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/stocks-and-indices/crsp-stock-and-indexes-version-2/crsp-ciz-faq/
 - CRSP Metadata Guide: https://wrds-www.wharton.upenn.edu/documents/1941/CRSP_METADATA_GUIDE_STOCK_INDEXES_FLAT_FILE_FORMAT_2_0_CIZ_09232022v.pdf

For more information about variables in CRSP, see:
https://wrds-www.wharton.upenn.edu/documents/396/CRSP_US_Stock_Indices_Data_Descriptions.pdf
I don't think this is updated for the new CIZ format, though.

Here is some information about the old SIZ CRSP format:
https://wrds-www.wharton.upenn.edu/documents/1095/CRSP_Flat_File_formats_and_notes.pdf

The following is an outdated programmer's guide to CRSP:
https://wrds-www.wharton.upenn.edu/documents/400/CRSP_Programmers_Guide.pdf


"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

import numpy as np
import pandas as pd
import wrds
from pandas.tseries.offsets import MonthEnd

from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))
DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


def get_CRSP_columns(wrds_username=WRDS_USERNAME, table_schema="crsp", table_name="msf_v2"):
    """Get all column names from CRSP monthly stock file (CIZ format)."""

    sql_query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = {table_schema}
        AND table_name = {table_name}
        ORDER BY ordinal_position;
    """
    
    db = wrds.Connection(wrds_username=wrds_username)
    columns = db.raw_sql(sql_query)
    db.close()
    
    return columns
 
# ==============================================================================================
# STOCK DATA
# ==============================================================================================

def pull_M_CRSP_stock(wrds_username=WRDS_USERNAME):
    """Pull necessary CRSP monthly stock data to
    compute Fama-French factors. Use the new CIZ format.

    Notes
    -----
    
    ## Cumulative Adjustment Factors (CFACPR and CFACSHR)
    https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/stocks-and-indices/crsp-stock-and-indexes-version-2/crsp-ciz-faq/

    In the legacy format, CRSP provided two data series, CFACPR and CFACSHR for
    cumulative adjustment factors for price and share respectively. In the new CIZ
    data format, these two data series are no longer provided, at least in the
    initial launch, per CRSP.

    WRDS understands the importance of these two variables to many researchers and
    we prepared a sample code that researchers can use to recreate the series using
    the raw adjustment factors. However, we need to caution users that the results
    of our sample code do not line up with the legacy CFACPR and CFACSHR completely.
    While it generates complete replication in 95% of the daily observations, we do
    observe major differences in the tail end. We do not have an explanation from
    CRSP about the remaining 5%, hence we urge researchers to use caution. Please
    contact CRSP support (Support@crsp.org) if you would like to discuss the issue
    of missing cumulative adjustment factors in the new CIZ data.

    For now, it's close enough to just let
    market_cap = mthprc * shrout

    """
    sql_query = """
        SELECT 
            permno, permco, mthcaldt, 
            issuertype, securitytype, securitysubtype, sharetype, 
            usincflg, 
            primaryexch, conditionaltype, tradingstatusflg,
            mthret, mthretx, shrout, mthprc
        FROM 
            crsp.msf_v2
        WHERE 
            mthcaldt >= '01/01/1959'
        """

    db = wrds.Connection(wrds_username=wrds_username)
    crsp_m = db.raw_sql(sql_query, date_cols=["mthcaldt"])
    db.close()

    # change variable format to int
    crsp_m[["permco", "permno"]] = crsp_m[["permco", "permno"]].astype(int)

    # Line up date to be end of month
    crsp_m["jdate"] = crsp_m["mthcaldt"] + MonthEnd(0)

    return crsp_m


def pull_D_CRSP_stock(wrds_username=WRDS_USERNAME):
    """Pull necessary CRSP monthly stock data to
    compute Fama-French factors. Use the new CIZ format.

    Notes
    -----
    
    ## Cumulative Adjustment Factors (CFACPR and CFACSHR)
    https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/stocks-and-indices/crsp-stock-and-indexes-version-2/crsp-ciz-faq/

    In the legacy format, CRSP provided two data series, CFACPR and CFACSHR for
    cumulative adjustment factors for price and share respectively. In the new CIZ
    data format, these two data series are no longer provided, at least in the
    initial launch, per CRSP.

    WRDS understands the importance of these two variables to many researchers and
    we prepared a sample code that researchers can use to recreate the series using
    the raw adjustment factors. However, we need to caution users that the results
    of our sample code do not line up with the legacy CFACPR and CFACSHR completely.
    While it generates complete replication in 95% of the daily observations, we do
    observe major differences in the tail end. We do not have an explanation from
    CRSP about the remaining 5%, hence we urge researchers to use caution. Please
    contact CRSP support (Support@crsp.org) if you would like to discuss the issue
    of missing cumulative adjustment factors in the new CIZ data.

    For now, it's close enough to just let
    market_cap = dlyprc * shrout

    """
    sql_query = """
        SELECT 
            permno, permco, dlycaldt, 
            issuertype, securitytype, securitysubtype, sharetype, 
            usincflg, 
            primaryexch, conditionaltype, tradingstatusflg,
            dlyret, dlyretx, shrout, dlyprc
        FROM 
            crsp.msf_v2
        WHERE 
            dlycaldt >= '01/01/1959'
        """

    db = wrds.Connection(wrds_username=wrds_username)
    crsp_d = db.raw_sql(sql_query, date_cols=["dlycaldt"])
    db.close()

    # change variable format to int
    crsp_d[["permco", "permno"]] = crsp_d[["permco", "permno"]].astype(int)

    # Line up date to be end of month
    crsp_d["jdate"] = crsp_d["dlycaldt"] + MonthEnd(0)

    return crsp_d


# ==============================================================================================
# INDEX DATA
# ==============================================================================================

def pull_CRSP_index_files(
    start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME
):
    """
    Pulls the CRSP index files from crsp_a_indexes.msix:
    (Monthly)NYSE/AMEX/NASDAQ Capitalization Deciles, Annual Rebalanced (msix)
    """
    # Pull index files
    query = f"""
        SELECT * 
        FROM crsp_a_indexes.msix
        WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
    """
    # with wrds.Connection(wrds_username=wrds_username) as db:
    #     df = db.raw_sql(query, date_cols=["month", "caldt"])
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["caldt"])
    db.close()
    return df


def pull_constituents(wrds_username=WRDS_USERNAME):
    db = wrds.Connection(wrds_username=wrds_username)

    df_constituents = db.raw_sql(""" 
    SELECT *
    from crsp_m_indexes.dsp500list_v2 
    """)

    # Convert string columns to datetime if they aren't already
    df_constituents["mbrstartdt"] = pd.to_datetime(df_constituents["mbrstartdt"])
    df_constituents["mbrenddt"] = pd.to_datetime(df_constituents["mbrenddt"])

    return df_constituents


# ==============================================================================================
# LOAD SAVED DATA
# ==============================================================================================

def load_CRSP_monthly_file(data_dir=DATA_DIR):
    path = Path(data_dir) / "CRSP_MSF_INDEX_INPUTS.parquet"
    df = pd.read_parquet(path)
    return df


def load_M_CRSP_stock(data_dir=DATA_DIR):
    path = Path(data_dir) / "CRSP_stock_m.parquet"
    crsp = pd.read_parquet(path)
    return crsp


def load_D_CRSP_stock(data_dir=DATA_DIR):
    path = Path(data_dir) / "CRSP_stock_d.parquet"
    crsp = pd.read_parquet(path)
    return crsp


def load_constituents(data_dir=DATA_DIR):
    return pd.read_parquet(data_dir / "df_sp500_constituents.parquet")


def _demo():
    crsp_d = load_D_CRSP_stock(data_dir=DATA_DIR)
    crsp_m = load_M_CRSP_stock(data_dir=DATA_DIR)
    df_msf = load_CRSP_monthly_file(data_dir=DATA_DIR)


if __name__ == "__main__":

    crsp_m = pull_M_CRSP_stock(wrds_username=WRDS_USERNAME)
    crsp_m.to_parquet(DATA_DIR / "CRSP_stock_m.parquet")

    crsp_d = pull_D_CRSP_stock(wrds_username=WRDS_USERNAME)
    crsp_d.to_parquet(DATA_DIR / "CRSP_stock_d.parquet")

    df_msix = pull_CRSP_index_files(start_date=START_DATE, end_date=END_DATE)
    df_msix.to_parquet(DATA_DIR / "CRSP_MSIX.parquet")

    constituents = pull_constituents(wrds_username=WRDS_USERNAME)
    constituents.to_parquet(DATA_DIR / "df_sp500_constituents.parquet")

