import datetime
import logging
import re
import sys
import zipfile
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

BASE_DIR = Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(Path(BASE_DIR) / 'src'))

# Local imports
import config


# =============================================================================
# Global Configuration
# =============================================================================
RAW_DATA_DIR = Path(config.RAW_DATA_DIR)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path(config.PROCESSED_DATA_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helper Functions (Caching, Reading/Writing Files)
# =============================================================================

def save_figure(
        fig: plt.Figure,
        plot_name_prefix: str,
) -> None:
    """
    Saves a matplotlib figure to a PNG file if save_plot is True.
    The filename pattern is "<prefix>_YYYYMMDD_HHMMSS.png".

    Parameters:
    fig (plt.Figure): The matplotlib figure to save.
    plot_name_prefix (str): The prefix for the plot filename.
    """
    filename  = f"{plot_name_prefix}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png"
    plot_path = OUTPUT_DIR / filename
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved to {plot_path}")


def cache_filename(
    code: str,
    filters_str: str,
    file_ext_list: List[str] = ["csv", "parquet", "zip"]
) -> List[Path]:
    """
    Generate a cache filename based on the code and filters,
    returning up to three paths: .csv, .parquet, and .zip.
    """
    # Simplify filter string
    if "date" not in filters_str:
        today_str = datetime.datetime.today().strftime('%Y%m%d')
        filters_str += f"_{today_str}"
        
    safe_filters_str = re.sub(
        r'export=[a-zA-Z]*|[^,]*=',
        '',
        filters_str
    ).replace("/", "_").replace("=", "_").replace(",", "_").replace("-", "")

    filenames = [
        f"{code.replace('/', '_')}__{safe_filters_str}.{file_ext}"
        for file_ext in file_ext_list
    ]
    # Clean up "empty" filters or duplicates
    filenames = [
        filename.replace("__.", ".")
                .replace("_.", ".")
        for filename in filenames
    ]

    return [RAW_DATA_DIR / filename for filename in filenames]


def file_cached(filepaths: List[Path]) -> Optional[Path]:
    """
    Check if any of the given filepaths exist (csv, parquet, or zip).
    Return the first that exists, else None.
    """
    for fp in filepaths:
        if Path(fp).exists():
            return Path(fp)
    return None


def read_cached_data(filepath: Path) -> pd.DataFrame:
    """
    Read cached data from a file, supporting various formats and compression types.
    """
    fmt = filepath.suffix.lstrip(".")

    if fmt == "csv":
        logging.info(f"Reading cached data from {filepath}")
        return pd.read_csv(filepath)

    elif fmt == "parquet":
        logging.info(f"Reading cached data from {filepath}")
        return pd.read_parquet(filepath)

    elif fmt == "zip":
        logging.info(f"Reading cached data from {filepath}")
        with zipfile.ZipFile(filepath, 'r') as z:
            file_name = z.namelist()[0]  # Assume only one file in the zip
            with z.open(file_name) as f:
                if file_name.endswith('.parquet'):
                    return pd.read_parquet(f)
                else:
                    return pd.read_csv(f)
    else:
        raise ValueError(f"Unsupported file format: {fmt}")


def write_cache_data(df: pd.DataFrame, filepath: Path, fmt: str = 'csv') -> None:
    """
    Write a DataFrame to file for caching.
    """
    if fmt == "parquet":
        df.to_parquet(filepath, index=False)
    else:
        df.to_csv(filepath, index=False)
    logging.info(f"Data cached to {filepath}")


def flatten_dict_to_str(d: Dict[str, Any]) -> str:
    """
    Recursively flatten a dict into a string representation for caching.
    Example:
       {'ticker': ['AAPL','MSFT'], 'date': {'gte': '2020-01-01'}} 
       -> "ticker=['AAPL','MSFT'],date.gte=2020-01-01"
    """
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            # recursively flatten sub-dict
            for subk, subv in v.items():
                items.append(f"{k}.{subk}={subv}")
        else:
            items.append(f"{k}={v}")
    return ",".join(items)

def read_excel_default(excel_name: str,
                       sheet_name: str = None, 
                       index_col : int = 0,
                       parse_dates: bool =True,
                       print_sheets: bool = False,
                       **kwargs):
    """
    Reads an Excel file and returns a DataFrame with specified options.

    Parameters:
    excel_name (str): The path to the Excel file.
    index_col (int, default=0): Column to use as the row index labels of the DataFrame.
    parse_dates (bool, default=True): Boolean to parse dates.
    print_sheets (bool, default=False): If True, prints the names and first few rows of all sheets.
    sheet_name (str or int, default=None): Name or index of the sheet to read. If None, reads the first sheet.
    **kwargs: Additional arguments passed to `pd.read_excel`.

    Returns:
    pd.DataFrame: DataFrame containing the data from the specified Excel sheet.

    Notes:
    - If `print_sheets` is True, the function will print the names and first few rows of all sheets and return None.
    - The function ensures that the index name is set to 'date' if the index column name is 'date', 'dates' or 'datatime', or if the index contains date-like values.
    """

    if print_sheets:
        excel_file = pd.ExcelFile(excel_name)  # Load the Excel file to get sheet names
        sheet_names = excel_file.sheet_names
        n = 0
        while True:
            try:
                sheet = pd.read_excel(excel_name, sheet_name=n)
                print(f'Sheet name: {sheet_names[n]}')
                print("Columns: " + ", ".join(list(sheet.columns)))
                print(sheet.head(3))
                n += 1
                print('-' * 70)
                print('\n')
            except:
                return
    sheet_name = 0 if sheet_name is None else sheet_name
    df = pd.read_excel(excel_name, index_col=index_col, parse_dates=parse_dates,  sheet_name=sheet_name, **kwargs)
    df.columns = [col.lower() for col in df.columns]
    if df.index.name is not None:
        if df.index.name in ['date', 'dates', 'datetime']:
            df.index.name = 'date'
    elif isinstance(df.index[0], (datetime.date, datetime.datetime)):
        df.index.name = 'date'
    return df


def read_csv_default(csv_name: str,
                     index_col: int = 0,
                     parse_dates: bool = True,
                     print_data: bool = False,
                     keep_cols: Union[List, str] = None,
                     drop_cols: Union[List, str] = None,
                     **kwargs):
    """
    Reads a CSV file and returns a DataFrame with specified options.

    Parameters:
    csv_name (str): The path to the CSV file.
    index_col (int, default=0): Column to use as the row index labels of the DataFrame.
    parse_dates (bool, default=True): Boolean to parse dates.
    print_data (bool, default=False): If True, prints the first few rows of the DataFrame.
    keep_cols (list or str, default=None): Columns to read from the CSV file.
    drop_cols (list or str, default=None): Columns to drop from the DataFrame.
    **kwargs: Additional arguments passed to `pd.read_csv`.

    Returns:
    pd.DataFrame: DataFrame containing the data from the CSV file.

    Notes:
    - The function ensures that the index name is set to 'date' if the index column name is 'date', 'dates' or 'datatime', or if the index contains date-like values.
    """

    df = pd.read_csv(csv_name, index_col=index_col, parse_dates=parse_dates, **kwargs)
    df.columns = [col.lower() for col in df.columns]

    # Filter columns if keep_cols is specified
    if keep_cols is not None:
        if isinstance(keep_cols, str):
            keep_cols = [keep_cols]
        df = df[keep_cols]

    # Drop columns if drop_cols is specified
    if drop_cols is not None:
        if isinstance(drop_cols, str):
            drop_cols = [drop_cols]
        df = df.drop(columns=drop_cols, errors='ignore')

    # Print data if print_data is True
    if print_data:
        print("Columns:", ", ".join(df.columns))
        print(df.head(3))
        print('-' * 70)
    
    # Set index name to 'date' if appropriate
    if df.index.name is not None:
        if df.index.name in ['date', 'dates', 'datetime']:
            df.index.name = 'date'
    elif isinstance(df.index[0], (datetime.date, datetime.datetime)):
        df.index.name = 'date'
    
    return df


# =============================================================================
# Business Day Utilities
# =============================================================================


def bday(input_date: Union[str, datetime.date, datetime.datetime]) -> bool:
    """
    Checks if a given date is a U.S. business day.

    Parameters:
    input_date (str or datetime.date or datetime.datetime): Date to check. If a string, should be in 'YYYY-MM-DD' format.

    Returns:
    bool: True if it is a business day, False otherwise.
    """
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    if isinstance(input_date, str):
        try:
            input_date = dt.strptime(input_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f'Invalid date string: {input_date}. Must be "YYYY-MM-DD".') from e

    # Generate a business day range from input_date to input_date
    return bool(len(pd.bdate_range(input_date, input_date, freq=us_bd)))


def prev_bday(
    input_date: Union[str, datetime.date, datetime.datetime],
    force_prev: bool = False
) -> Union[str, datetime.date, datetime.datetime]:
    """
    Finds the previous U.S. business day from the given date.

    Parameters:
    input_date (str or datetime.date or datetime.datetime): Date from which to find the previous business day. If a string, should be in 'YYYY-MM-DD' format.
    force_prev (bool, default=False): If True, forces the function to move at least one business day back, even if 'input_date' is already a business day.

    Returns:
    (str or datetime.date or datetime.datetime): The previous business day in the same format the input was provided.
    """
    date_str = False
    if isinstance(input_date, str):
        date_str = True
        try:
            input_date = dt.strptime(input_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f'Invalid date string: {input_date}. Must be "YYYY-MM-DD".') from e

    if force_prev:
        input_date -= timedelta(days=1)

    while not bday(input_date):
        input_date -= timedelta(days=1)

    if date_str:
        return input_date.strftime('%Y-%m-%d')
    return input_date


def next_business_day(input_date: Union[str, datetime.date, datetime.datetime]) -> datetime.date:
    """
    Finds the next U.S. business day after the given date, including the possibility that 'input_date' is itself not a business day.

    Parameters:
    input_date (str or datetime.date or datetime.datetime): Date from which to find the next business day. If a string, should be in 'YYYY-MM-DD' format.

    Returns:
    datetime.date: The next business day (as a date).
    """
    if isinstance(input_date, str):
        try:
            input_date = dt.strptime(input_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f'Invalid date string: {input_date}. Must be "YYYY-MM-DD".') from e

    one_day = timedelta(days=1)
    us_holidays = holidays.US()

    next_day = input_date
    while (next_day.weekday() in holidays.WEEKEND) or (next_day in us_holidays):
        next_day += one_day

    return next_day.date()


# =============================================================================
# Manipulating DataFrames Utilities
# =============================================================================

def time_series_to_df(returns: Union[pd.DataFrame, pd.Series, List[pd.Series]], name: str = "Returns"):
    """
    Converts returns to a DataFrame if it is a Series or a list of Series.

    Parameters:
    returns (pd.DataFrame, pd.Series or List or pd.Series): Time series of returns.

    Returns:
    pd.DataFrame: DataFrame of returns.
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.copy()
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    elif isinstance(returns, list):
        returns_list = returns.copy()
        returns = pd.DataFrame({})

        for series in returns_list:
            if isinstance(series, pd.Series):
                returns = returns.merge(series, right_index=True, left_index=True, how='outer')
            else:
                raise TypeError(f'{name} must be either a pd.DataFrame or a list of pd.Series')
            
    # Convert returns to float
    try:
        returns = returns.apply(lambda x: x.astype(float))
    except ValueError:
        print(f'Could not convert {name} to float. Check if there are any non-numeric values')
        pass

    return returns


def fix_dates_index(returns: pd.DataFrame):
    """
    Fixes the date index of a DataFrame if it is not in datetime format and convert returns to float.

    Parameters:
    returns (pd.DataFrame): DataFrame of returns.

    Returns:
    pd.DataFrame: DataFrame with datetime index.
    """
    # Check if 'date' is in the columns and set it as the index

    # Set index name to 'date' if appropriate
    
    if returns.index.name is not None:
        if returns.index.name.lower() in ['date', 'dates', 'datetime']:
            returns.index.name = 'date'
    elif isinstance(returns.index[0], (datetime.date, datetime.datetime)):
        returns.index.name = 'date'
    elif 'date' in returns.columns.str.lower():
        returns = returns.rename({'Date': 'date'}, axis=1)
        returns = returns.set_index('date')
    elif 'datetime' in returns.columns.str.lower():
        returns = returns.rename({'Datetime': 'date'}, axis=1)
        returns = returns.rename({'datetime': 'date'}, axis=1)
        returns = returns.set_index('date')

    # Convert dates to datetime if not already in datetime format or if minutes are 0
    try:
        returns.index = pd.to_datetime(returns.index, utc=True)
    except ValueError:
        print('Could not convert the index to datetime. Check the index format for invalid dates.')
    if not isinstance(returns.index, pd.DatetimeIndex) or (returns.index.minute == 0).all():
        returns.index = pd.to_datetime(returns.index.map(lambda x: x.date()))
        
    # Convert returns to float
    try:
        returns = returns.apply(lambda x: x.astype(float))
    except ValueError:
        print('Could not convert returns to float. Check if there are any non-numeric values')
        pass

    return returns


def filter_columns_and_indexes(
    df: pd.DataFrame,
    keep_columns: Union[list, str],
    drop_columns: Union[list, str],
    keep_indexes: Union[list, str],
    drop_indexes: Union[list, str]
):
    """
    Filters a DataFrame based on specified columns and indexes.

    Parameters:
    df (pd.DataFrame): DataFrame to be filtered.
    keep_columns (list or str): Columns to keep in the DataFrame.
    drop_columns (list or str): Columns to drop from the DataFrame.
    keep_indexes (list or str): Indexes to keep in the DataFrame.
    drop_indexes (list or str): Indexes to drop from the DataFrame.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """

    if not isinstance(df, (pd.DataFrame, pd.Series)):
        return df
    
    df = df.copy()

    # Columns
    if keep_columns is not None:
        keep_columns = [re.escape(col) for col in keep_columns]
        keep_columns = "(?i).*(" + "|".join(keep_columns) + ").*" if isinstance(keep_columns, list) else "(?i).*" + keep_columns + ".*"
        df = df.filter(regex=keep_columns)
        if drop_columns is not None:
            print('Both "keep_columns" and "drop_columns" were specified. "drop_columns" will be ignored.')

    elif drop_columns is not None:
        drop_columns = [re.escape(col) for col in drop_columns]
        drop_columns = "(?i).*(" + "|".join(drop_columns) + ").*" if isinstance(drop_columns, list) else "(?i).*" + drop_columns + ".*"
        df = df.drop(columns=df.filter(regex=drop_columns).columns)

    # Indexes
    if keep_indexes is not None:
        keep_indexes = [re.escape(col) for col in keep_indexes]
        keep_indexes = "(?i).*(" + "|".join(keep_indexes) + ").*" if isinstance(keep_indexes, list) else "(?i).*" + keep_indexes + ".*"
        df = df.filter(regex=keep_indexes, axis=0)
        if drop_indexes is not None:
            print('Both "keep_indexes" and "drop_indexes" were specified. "drop_indexes" will be ignored.')

    elif drop_indexes is not None:
        drop_indexes = [re.escape(col) for col in drop_indexes]
        drop_indexes = "(?i).*(" + "|".join(drop_indexes) + ").*" if isinstance(drop_indexes, list) else "(?i).*" + drop_indexes + ".*"
        df = df.filter(regex=keep_indexes, axis=0)
    
    return df