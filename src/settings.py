"""
blank_project/settings.py

Load project configurations from .env files.
Provides easy access to paths and credentials used in the project.
Meant to be used as an imported module.

If `config.py` is run on its own, it will create the appropriate
directories.

For information about the rationale behind decouple and this module,
see https://pypi.org/project/python-decouple/

Note that decouple mentions that it will help to ensure that
the project has "only one configuration module to rule all your instances."
This is achieved by putting all the configuration into the `.env` file.
You can have different sets of variables for difference instances, 
such as `.env.development` or `.env.production`. You would only
need to copy over the settings from one into `.env` to switch
over to the other configuration, for example.

"""

from decouple import config as _config
from pathlib import Path
from platform import system
from pandas import to_datetime

try:
    from scheffer_quant import settings as sq_settings
    _HAS_SCHEFFER_QUANT = True
except ImportError:
    # scheffer_quant not installed or unavailable.
    sq_settings = None
    _HAS_SCHEFFER_QUANT = False


###############################################################################
# Define helper(s)
###############################################################################

def get_os():
    os_name = system()
    if os_name == "Windows":
        return "windows"
    elif os_name == "Darwin":
        return "nix"
    elif os_name == "Linux":
        return "nix"
    else:
        return "unknown"
    
# Absolute path to root directory of the project
BASE_DIR = Path(__file__).absolute().parent.parent

def if_relative_make_abs(path: Path|str, base: Path = BASE_DIR) -> Path:
    """
    If `path` is relative, interpret it as relative to `base`.
    Return an absolute, resolved Path.
    """
    path = Path(path)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


###############################################################################
# Load environment variables
###############################################################################

OS_TYPE = get_os()

# fmt: off
## Other .env variables
WRDS_USERNAME = _config("WRDS_USERNAME", default="")
NASDAQ_API_KEY = _config("NASDAQ_API_KEY", default="")
START_DATE = _config("START_DATE", default="1913-01-01", cast=to_datetime)
END_DATE = _config("END_DATE", default="2024-12-31", cast=to_datetime)
USER = _config("USER", default="")

## Paths
DATA_DIR = if_relative_make_abs(_config('DATA_DIR', default=Path('_data'), cast=Path))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MANUAL_DATA_DIR = if_relative_make_abs(_config('DATA_MANUAL_DIR', default=DATA_DIR / "manual", cast=Path))
LOG_DIR = if_relative_make_abs(_config('LOG_DIR', default=Path('logs'), cast=Path))
OUTPUT_DIR = if_relative_make_abs(_config('OUTPUT_DIR', default=Path('_output'), cast=Path))
PUBLISH_DIR = if_relative_make_abs(_config('PUBLISH_DIR', default=Path('docs'), cast=Path))
REPORTS_DIR = if_relative_make_abs(_config('REPORTS_DIR', default=Path('reports'), cast=Path))

PLOT_WIDTH = _config("PLOT_WIDTH", default=10)
PLOT_HEIGHT = _config("PLOT_HEIGHT", default=6)

###############################################################################
# Update scheffer_quant.settings.Config if available
###############################################################################

def _try_update_scheffer_quant_config():
    """
    If scheffer_quant is installed and we want to use it,
    update its class-based config with the values we just loaded.
    """
    if not _HAS_SCHEFFER_QUANT:
        return

    # We call the .update() method from scheffer_quant/settings.py
    sq_settings.Config.update(
        BASE_DIR=BASE_DIR,
        DATA_DIR=DATA_DIR,
        RAW_DATA_DIR=RAW_DATA_DIR,
        MANUAL_DATA_DIR=MANUAL_DATA_DIR,
        LOG_DIR=LOG_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        WRDS_USERNAME=WRDS_USERNAME,
        NASDAQ_API_KEY=NASDAQ_API_KEY,
        START_DATE=START_DATE,
        END_DATE=END_DATE,
        PLOT_HEIGHT=PLOT_HEIGHT,
        PLOT_WIDTH=PLOT_WIDTH,
    )

_try_update_scheffer_quant_config()

###############################################################################
# 4) Provide a create_dirs() helper
###############################################################################

def create_dirs():
    """
    Create the typical directory structure needed for your blank_project.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    # etc.

if __name__ == "__main__":
    create_dirs()