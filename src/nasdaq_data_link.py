"""
nasdaq_data_link.py

Utility module to interact with the Nasdaq Data Link (Quandl) Python client,
focusing on the OptionWorks Futures Options (OWF) product: AR/IVM (Implied
Volatility Model) and AR/IVS (Implied Volatility Surfaces).

This version moves local caching into the general Public API functions.

"""

import logging
import datetime
import zipfile
from typing import Optional, Dict, Any, List, Union

import nasdaqdatalink as ndl
import pandas as pd

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(Path(BASE_DIR) / 'src'))

# Local imports
import config
from data_utils import *

# =============================================================================
# Set up logging
# =============================================================================
LOG_PATH = Path(config.LOG_DIR) / "nasdaq_data_link.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# =============================================================================
# Global Configuration
# =============================================================================
ndl.ApiConfig.api_key = config.NASDAQ_API_KEY

RAW_DATA_DIR = Path(config.RAW_DATA_DIR)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Public API Functions (with caching)
# =============================================================================

def do_bulk_download(
    database_code: str,
    filename: Optional[str] = None,
    download_type: Optional[str] = None
) -> str:
    """
    Downloads an entire database (or partial) as a .zip file from Nasdaq Data Link.
    No DataFrame result, so no local caching applied here by default.
    """
     # 1) Build a cache key from dataset_code + kwargs
    filter_str = flatten_dict_to_str(kwargs) if kwargs else "no_filters"
    cache_paths = cache_filename(database_code, filter_str)

    # 2) If cached file exists, load
    cached_fp = file_cached(cache_paths)
    if cached_fp:
        return read_cached_data(cached_fp)

    # 3) Otherwise fetch from the API
    logging.info(f"Initiating bulk download for database '{database_code}'")
    kwargs = {}
    if filename:
        kwargs['filename'] = filename
    if download_type:
        kwargs['download_type'] = download_type
    downloaded_file = ndl.bulkdownload(database_code, **kwargs)
    logging.info(f"Bulk download completed. Saved to {downloaded_file}")
    return downloaded_file


def fetch_dataset_data(dataset_code: str, export_zip: bool = False, **kwargs) -> pd.DataFrame:
    ...
    filter_str = flatten_dict_to_str(kwargs) if kwargs else "no_filters"
    cache_paths = cache_filename(dataset_code, filter_str)

    # Attempt to read from cache
    cached_fp = file_cached(cache_paths)
    if cached_fp:
        return read_cached_data(cached_fp)

    # Otherwise, fetch from API
    logging.info(f"Fetching dataset '{dataset_code}' with params: {kwargs}")
    df = ndl.get(dataset_code, **kwargs)
    logging.info(f"Retrieved {len(df)} rows from dataset '{dataset_code}'")

    # Cache if not empty
    if not df.empty:
        zip_path = next((path for path in cache_paths if path.endswith('.zip')), None)
        csv_path = next((path for path in cache_paths if path.endswith('.csv')), None)
        csv_file_name = csv_path.name

        if export_zip:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                csv_data = df.to_csv(index=False)
                zipf.writestr(csv_file_name, csv_data)
            file_size = os.path.getsize(zip_path)
            logging.info(f"Cached data as ZIP to {zip_path} (size: {file_size} bytes)")
        else:
            write_cache_data(df, csv_path, fmt='csv')
            file_size = os.path.getsize(csv_path)
            logging.info(f"Cached data to {csv_path} (size: {file_size} bytes)")

    return df


def fetch_datatable_data(
    table_code: str,
    filters: Optional[Dict[str, Any]] = None,
    qopts: Optional[Dict[str, Any]] = None,
    paginate: bool = False
) -> pd.DataFrame:
    """
    Generic function to retrieve data from a Nasdaq Data Link datatable (e.g. 'ZACKS/FC'),
    with caching built in. Supports advanced filters, qopts, and optional auto-pagination.

    :param table_code: The datatable code, e.g. 'ZACKS/FC' or 'AR/IVM'.
    :param filters: A dict of filters, e.g. {'exchange_code': 'NYM', 'date': {'gte': '2020-01-01'}}.
    :param qopts: Additional params, e.g. {'columns': [...], 'export': 'zip', 'per_page': 10000}.
    :param paginate: Whether to auto-paginate or just fetch a single page.
    :return: A pandas DataFrame of the requested data.
    """
    filters = filters or {}
    qopts = qopts or {}

    # 1) Build cache key
    # Flatten both filters and qopts into a single string
    combined_str = flatten_dict_to_str({**filters, **qopts})

    cache_paths = cache_filename(table_code, combined_str)
    cached_fp = file_cached(cache_paths)
    if cached_fp:
        return read_cached_data(cached_fp)

    # 2) Fetch from nasdaqdatalink
    logging.info(f"Fetching datatable '{table_code}' with filters={filters}, qopts={qopts}, paginate={paginate}")
    if paginate:
        df = ndl.get_table(table_code, paginate=True, qopts=qopts, **filters)
    else:
        df = ndl.get_table(table_code, qopts=qopts, **filters)
    logging.info(f"Returned {len(df)} rows from '{table_code}'")

    # 3) Cache the result if not empty
    if not df.empty:
        csv_path = next((path for path in cache_paths if str(path).endswith('.csv')), None)
        zip_path = next((path for path in cache_paths if str(path).endswith('.zip')), None)
        csv_file_name = csv_path.name
        
        if not zip_path or not csv_path:
            raise ValueError("Cache paths must include both a .zip and a .csv path.")
        
        if qopts.get("export") == "zip":
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                csv_data = df.to_csv(index=False)
                zipf.writestr(csv_file_name, csv_data)
            file_size = os.path.getsize(zip_path)
            logging.info(f"Cached data as ZIP to {zip_path} (size: {file_size} bytes)")
        else:
            write_cache_data(df, csv_path, fmt='csv')
            file_size = os.path.getsize(csv_path)
            logging.info(f"Cached data to {csv_path} (size: {file_size} bytes)")

    return df


def fetch_point_in_time(
    table_code: str,
    interval: str,
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Fetch point-in-time data from Nasdaq Data Link for a specific date or a date range,
    with caching support.

    :param table_code: e.g. 'DATABASE/CODE'.
    :param interval: 'asofdate', 'from', or 'between'.
    :param date: The specific date for interval='asofdate' (YYYY-MM-DD or full ISO8601).
    :param start_date: For 'from' or 'between' interval.
    :param end_date: For 'from' or 'between' interval.
    :param kwargs: Additional filter or qopts params (these get appended to cache).
    :return: A pandas DataFrame of point-in-time data.
    """
    # 1) Build cache key
    pit_params = {
        'interval': interval,
        'date': date,
        'start_date': start_date,
        'end_date': end_date,
        **kwargs
    }
    filter_str = flatten_dict_to_str(pit_params)
    cache_paths = cache_filename(table_code, filter_str)
    cached_fp = file_cached(cache_paths)
    if cached_fp:
        return read_cached_data(cached_fp)

    # 2) Fetch from Nasdaq Data Link
    logging.info(f"Fetching point-in-time data from '{table_code}' interval='{interval}', date={date}, "
                 f"start_date={start_date}, end_date={end_date}, extra={kwargs}")
    df = ndl.get_point_in_time(
        table_code,
        interval=interval,
        date=date,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )
    logging.info(f"Retrieved {len(df)} rows from point-in-time endpoint.")

    # 3) Cache if not empty
    if not df.empty:
        write_cache_data(df, cache_paths[0], fmt='csv')
        # Log the file size after caching
        file_size = os.path.getsize(cache_paths[0])
        logging.info(f"Cached data to {cache_paths[0]} (size: {file_size} bytes)")

    return df


def fetch_table_metadata(table_code: str) -> Dict[str, Any]:
    """
    Fetch metadata for a particular table (e.g., AR/IVM or AR/IVS) from Nasdaq Data Link.
    (No caching necessary, as metadata is typically small and quick to fetch.)
    """
    logging.info(f"Fetching metadata for table code: {table_code}")
    import requests

    url = f"https://data.nasdaq.com/api/v3/datatables/{table_code.replace('/', '/')}/metadata"
    params = {"api_key": config.NASDAQ_API_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()

    metadata = response.json()
    logging.info(f"Metadata retrieved successfully for {table_code}")
    return metadata


# =============================================================================
# Specialized AR/IVM function (NO caching here, calls fetch_datatable_data instead)
# =============================================================================
def get_owf_data(
    table_code: str,
    date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange_code: Optional[str] = None,
    futures_code: Optional[str] = None,
    option_code: Optional[str] = None,
    expiration: Optional[str] = None,
    additional_filters: Optional[Dict[str, Any]] = None,
    columns: Optional[Union[str, List[str]]] = None,
    export_zip: bool = False,
    per_page: Optional[int] = None,
    cursor_id: Optional[str] = None,
    paginate: bool = False
) -> pd.DataFrame:
    """
    Specialized wrapper to retrieve OptionWorks Futures Options data (AR/IVM or AR/IVS).
    No direct caching is done here; we delegate caching to fetch_datatable_data.

    Documentation: https://data.nasdaq.com/databases/OWF#documentation

    :param table_code: e.g., 'AR/IVM' or 'AR/IVS'
    :param date: A single date (e.g., '2023-01-01').
    :param start_date: e.g., '2020-01-01'
    :param end_date: e.g., '2024-12-31'
    :param exchange_code: Exchange filter
    :param futures_code: Futures code filter
    :param option_code: Option code filter
    :param expiration: e.g., 'Z2024' or '2024-12-31'
    :param additional_filters: Additional dict-based filters
    :param columns: List[str] of columns
    :param export_zip: if True => qopts["export"] = "zip"
    :param per_page: e.g. 10000
    :param cursor_id: for manual pagination
    :param paginate: if True => auto-pagination
    :return: DataFrame with results
    """
    # Build filters
    filters = {}

    if exchange_code:
        filters["exchange_code"] = exchange_code
    if futures_code:
        filters["futures_code"] = futures_code
    if option_code:
        filters["option_code"] = option_code
    if expiration:
        filters["expiration"] = expiration
    if date:
        filters["date"] = date  # Single date
    else:
        # Handle date range if no specific date is given
        date_filter = {}
        if start_date and not end_date:
            date_filter["gte"] = start_date
            date_filter["lte"] = datetime.datetime.today().strftime('%Y-%m-%d')
        elif not start_date and end_date:
            date_filter["gte"] = pd.Timestamp('2000-01-01').strftime('%Y-%m-%d')
            date_filter["lte"] = end_date
        else:
            date_filter["gte"] = start_date
            date_filter["lte"] = end_date
        filters["date"] = date_filter

    if additional_filters:
        filters.update(additional_filters)

    # Build qopts
    qopts = {}
    if columns:
        qopts["columns"] = columns if isinstance(columns, list) else [columns]
    if export_zip:
        qopts["export"] = "zip"
    if per_page:
        qopts["per_page"] = per_page
    if cursor_id:
        qopts["cursor_id"] = cursor_id

    logging.info(f"Requesting {table_code} from Nasdaq with filters: {filters} and qopts: {qopts}")

    # Delegate to the generic fetch_datatable_data, which handles caching
    try:
        df = fetch_datatable_data(
            table_code=table_code,
            filters=filters,
            qopts=qopts,
            paginate=paginate
        )
    except Exception as e:
        logging.error(f"Error fetching data from Nasdaq Data Link: {e}")
        raise

    return df


# =============================================================================
# Example usage
# =============================================================================
if __name__ == '__main__':
    # Example: specialized AR/IVM call
    '''
    df_ivm = get_owf_data(
        table_code='AR/IVM',
        exchange_code='NYM',
        futures_code='RB',
        start_date='2020-01-01',
        end_date='2024-12-31',
        columns=['exchange_code', 'futures_code', 'expiration','date','atm'],
        export_zip=True,
        paginate=False
    )
    '''

    START_DATE = '2021-12-03'
    END_DATE = '2024-08-31'

    futures_list = [('CBT', 'TU'), ('CBT', 'TY'), ('NYM', 'RB'), ('ICE', 'G')]
    futures_data = pd.DataFrame()

    for exchange, futures_code in futures_list:
        data = get_owf_data(
                    table_code = 'AR/IVM',
                    exchange_code=exchange,
                    futures_code=futures_code,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    export_zip=True,
                    paginate=True
                    )
        futures_data = pd.concat([futures_data, data], axis=0)

    print(data.head())
    print(data.info())
    print(data.describe())

    # # Example: Bulk download entire database
    # # do_bulk_download('EOD', filename='EOD_DB.zip', download_type='partial')

    # # Example: Fetch a single dataset with caching
    # # df_oil = fetch_dataset_data('NSE/OIL', start_date='2010-01-01', end_date='2014-01-01')
    # # print(df_oil.head())

    # # Example: Generic datatable usage with caching
    # # df_zacks = fetch_datatable_data(
    # #     'ZACKS/FC',
    # #     filters={'ticker': ['AAPL','MSFT'], 'per_end_date': {'gte':'2020-01-01'}},
    # #     qopts={'columns': ['ticker','per_end_date','comp_name']},
    # #     paginate=True
    # # )
    # # print(df_zacks.head())

    # # Example: Point-in-time usage with caching
    # # df_pit = fetch_point_in_time('DATABASE/CODE', interval='asofdate', date='2020-01-01')
    # # print(df_pit.head())
