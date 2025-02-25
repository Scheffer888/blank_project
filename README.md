Example Project Using the ChartBook Template
=============================================

## About this project

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## Quick Start

To quickest way to run code in this repo is to use the following steps. First, you must have the `conda`  
package manager installed (e.g., via Anaconda). However, I recommend using `mamba`, via [miniforge]
(https://github.com/conda-forge/miniforge) as it is faster and more lightweight than `conda`. Second, you 
must have TexLive (or another LaTeX distribution) installed on your computer and available in your path.
You can do this by downloading and 
installing it from here ([windows](https://tug.org/texlive/windows.html#install) 
and [mac](https://tug.org/mactex/mactex-download.html) installers).
Having done these things, open a terminal and navigate to the root directory of the project and create a 
conda environment using the following command:
```
conda create -n blank python=3.12
conda activate blank
```
and then install the dependencies with pip
```
pip install -r requirements.txt
```
Finally, you can then run 
```
doit
```
And that's it!

If you would also like to run the R code included in this project, you can either install
R and the required packages manually, or you can use the included `environment.yml` file.
To do this, run
```
mamba env create -f environment.yml
```
I'm using `mamba` here because `conda` is too slow. Activate the environment. 
Then, make sure to uncomment
out the RMarkdown task from the `dodo.py` file. Then,
run `doit` as before.

### Other commands

#### Unit Tests and Doc Tests

You can run the unit test, including doctests, with the following command:
```
pytest --doctest-modules
```
You can build the documentation with:
```
rm ./src/.pytest_cache/README.md 
jupyter-book build -W ./
```
Use `del` instead of rm on Windows

#### Setting Environment Variables

You can 
[export your environment variables](https://stackoverflow.com/questions/43267413/how-to-set-environment-variables-from-env-file) 
from your `.env` files like so, if you wish. This can be done easily in a Linux or Mac terminal with the following command:
```
set -a ## automatically export all variables
source .env
set +a
```
In Windows, this can be done with the included `set_env.bat` file,
```
set_env.bat
```

### General Directory Structure

 - Folders that start with `_` are automatically generated. The entire folder should be able to be deleted, because the code can be run again, which would again generate all of the contents. 

 - Anything in the `_data` folder (or your own RAW_DATA_DIR) or in the `_output` folder should be able to be recreated by running the code and can safely be deleted.

 - The `assets` folder is used for things like hand-drawn figures or other pictures that were not generated from code. These things cannot be easily recreated if they are deleted.

 - `_output` contains the .py generated from jupyter notebooks, and the jupyter notebooks with outputs, both in .md and in .html
 
 - `/src` contains the actual code. All notebooks in this folder will be stored cleaned from outputs (after running doit). That is in order to avoid unecessary commits from changes from simply opening or running the notebook.

 - The `data_manual` (DATA_MANUAL_DIR) is for data that cannot be easily recreated. 

 - `doit` Python module is used as the task runner. It works like `make` and the associated `Makefile`s. To rerun the code, install `doit` (https://pydoit.org/) and execute the command `doit` from the `src` directory. Note that doit is very flexible and can be used to run code commands from the command prompt, thus making it suitable for projects that use scripts written in multiple different programming languages.

 - `.env` file is the container for absolute paths that are private to each collaborator in the project. You can also use it for private credentials, if needed. It should not be tracked in Git.

#### Sphinx:
 - _docs is where the run notebooks are put along with the copies of the docs source code, it is a directory where the html build happens. All of this is done automatically with doit. Then, the build html gets put into docs since GitHub pages requires that the HTML be kept in a directory called docs_src

- Sphinx documentation is saved on `/docs` and `/docs_src` folders. These folder are, by default, ignored by Git. If you want to publish the documentation, you need to create a new branch "gs-pages" and uncomment those from .gitignore.

- `.nojekyll`: when you publish the documentation to GitHub pages, if you are not using the statis Jekll theme, you need to add the nojekyll file to the root of the docs folder. This is because GitHub pages uses Jekyll by default, and it will ignore any files that start with an underscore. Since we are using Sphinx, we need to add this file to prevent GitHub pages from ignoring our files.
- MyST: a flavor/dialect to markdown that has a lot of features from Restructured Text, that is more feature-rich, which Sphinx use.

#### Specific Files

- `pipeline.json`: is currently orchestrating the Chartbook, which will replace Sphinx. It will later be moved to pyproject.yoml.
- `dodo.py`: is the file that defines the tasks to be run by doit. It is the equivalent of a Makefile. It is the main entry point for running code in this project. It is also used to generate the documentation.
- `settings.py`: is the file that defines the settings for the project. It is the main entry point for running code in this project. It is also used to generate the documentation.
- `pyproject.toml`: is currently used to configure some 
- 



### Data and Output Storage

I'll often use a separate folder for storing data. Any data in the data folder can be deleted and recreated by rerunning the PyDoit command (the pulls are in the dodo.py file). Any data that cannot be automatically recreated should be stored in the "data_manual" folder (or DATA_MANUAL_DIR).

Because of the risk of manually-created data getting changed or lost, I prefer to keep it under version control if I can.

Thus, data in the "_data" folder is excluded from Git (see the .gitignore file), while the "data_manual" folder is tracked by Git.

Output is stored in the "_output" directory. This includes dataframes, charts, and rendered notebooks. When the output is small enough, you can have it under version control to keep track of how dataframes change as analysis progresses, for example.

The _data directory and _output directory can be kept elsewhere on the machine. To make this easy, I always include the ability to customize these locations by defining the path to these directories in environment variables, which I intend to be defined in the `.env` file, though they can also simply be defined on the command line or elsewhere. The `settings.py` is reponsible for loading these environment variables and doing some like preprocessing on them.

The `settings.py` file is the entry point for all other scripts to these definitions. That is, all code that references these variables and others are loading by importing `config`.

### Naming Conventions

 - **`pull_` vs `load_`**: Files or functions that pull data from an external  data source are prepended with "pull_", as in "pull_fred.py". Functions that load data that has been cached in the "_data" folder are prepended with "load_".
 For example, inside of the `pull_CRSP_Compustat.py` file there is both a
 `pull_compustat` function and a `load_compustat` function. The first pulls from
 the web, whereas the other loads cached data from the "_data" directory.


### Dependencies and Virtual Environments

#### Working with `pip` requirements

`conda` allows for a lot of flexibility, but can often be slow. `pip`, however, is fast for what it does.  You can install the requirements for this project using the `requirements.txt` file specified here. Do this with the following command:
```
pip install -r requirements.txt
```

The requirements file can be created like this:
```
pip list --format=freeze
```

#### Working with `conda` environments

The dependencies used in this environment (along with many other environments commonly used in data science) are stored in the conda environment called `blank` which is saved in the file called `environment.yml`. To create the environment from the file (as a prerequisite to loading the environment), use the following command:

```
conda env create -f environment.yml
```

Now, to load the environment, use

```
conda activate blank
```

Note that an environment file can be created with the following command:

```
conda env export > environment.yml
```

However, it's often preferable to create an environment file manually, as was done with the file in this project.

Also, these dependencies are also saved in `requirements.txt` for those that would rather use pip. Also, GitHub actions work better with pip, so it's nice to also have the dependencies listed here. This file is created with the following command:

```
pip freeze > requirements.txt
```

**Other helpful `conda` commands**

- Create conda environment from file: `conda env create -f environment.yml`
- Activate environment for this project: `conda activate blank`
- Remove conda environment: `conda remove --name blank --all`
- Create blank conda environment: `conda create --name myenv --no-default-packages`
- Create blank conda environment with different version of Python: `conda create --name myenv --no-default-packages python` Note that the addition of "python" will install the most up-to-date version of Python. Without this, it may use the system version of Python, which will likely have some packages installed already.

#### `mamba` and `conda` performance issues

Since `conda` has so many performance issues, it's recommended to use `mamba` instead. I recommend installing the `miniforge` distribution. See here: https://github.com/conda-forge/miniforge

