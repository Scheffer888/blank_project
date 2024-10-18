"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based

"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

from os import getcwd
from os import path
from os import environ
import shutil

## Custom reporter: Print PyDoit Text in Green
# This is helpful because some tasks write to sterr and pollute the output in
# the console. I don't want to mute this output, because this can sometimes
# cause issues when, for example, LaTeX hangs on an error and requires
# presses on the keyboard before continuing. However, I want to be able
# to easily see the task lines printed by PyDoit. I want them to stand out
# from among all the other lines printed to the console.
from doit.reporter import ConsoleReporter
from colorama import Fore, Style, init


class GreenReporter(ConsoleReporter):
    def write(self, stuff, **kwargs):
        doit_mark = stuff.split(" ")[0].ljust(2)
        task = " ".join(stuff.split(" ")[1:]).strip() + "\n"
        output = (
            Fore.GREEN
            + doit_mark
            + f" {path.basename(getcwd())}: "
            + task
            + Style.RESET_ALL
        )
        self.outstream.write(output)


DOIT_CONFIG = {
    "reporter": GreenReporter,
    # other config here...
    # "cleanforget": True, # Doit will forget about tasks that have been cleaned.
}
init(autoreset=True)

import config
from pathlib import Path
from doit.tools import run_once
import pipeline_publish

BASE_DIR = Path(config.BASE_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)
DOCS_PUBLISH_DIR = Path(config.DOCS_PUBLISH_DIR)
OS_TYPE = config.OS_TYPE

## Helpers for handling Jupyter Notebook tasks
# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
def jupyter_execute_notebook(notebook):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --log-level WARN --inplace ./src/{notebook}.ipynb"
def jupyter_to_html(notebook, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --log-level WARN --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_md(notebook, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --log-level WARN --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_python(notebook, build_dir):
    """Convert a notebook to a python script"""
    return f"jupyter nbconvert --log-level WARN --to python ./src/{notebook}.ipynb --output _{notebook}.py --output-dir {build_dir}"
def jupyter_clear_output(notebook):
    return f"jupyter nbconvert --log-level WARN --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace ./src/{notebook}.ipynb"
# fmt: on


def copy_notebook_to_folder(notebook_stem, origin_folder, destination_folder):
    origin_path = Path(origin_folder) / f"{notebook_stem}.ipynb"
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination_path = destination_folder / f"{notebook_stem}.ipynb"
    if OS_TYPE == "nix":
        command = f"cp {origin_path} {destination_path}"
    else:
        command = f"copy  {origin_path} {destination_path}"
    return command


##################################
## Begin rest of PyDoit tasks here
##################################

def task_pull_fred():
    """ """
    file_dep = [
        "./src/pull_fred.py",
        "./src/pull_ofr_api_data.py",
    ]
    targets = [
        DATA_DIR / "fred.parquet",
        DATA_DIR / "ofr_public_repo_data.parquet",
    ]

    return {
        "actions": [
            "ipython ./src/config.py",
            "ipython ./src/pull_fred.py",
            "ipython ./src/pull_ofr_api_data.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": [],  # Don't clean these files by default. The ideas
        # is that a data pull might be expensive, so we don't want to
        # redo it unless we really mean it. So, when you run
        # doit clean, all other tasks will have their targets
        # cleaned and will thus be rerun the next time you call doit.
        # But this one wont.
        # Use doit forget --all to redo all tasks. Use doit clean
        # to clean and forget the cheaper tasks.
    }


##############################$
## Demo: Other misc. data pulls
##############################$
# def task_pull_all():
#     """ """
#     file_dep = [
#         "./src/pull_bloomberg.py",
#         "./src/pull_CRSP_Compustat.py",
#         "./src/pull_CRSP_stock.py",
#         "./src/pull_fed_yield_curve.py",
#         ]
#     file_output = [
#         "bloomberg.parquet",
#         "CRSP_Compustat.parquet",
#         "CRSP_stock.parquet",
#         "fed_yield_curve.parquet",
#         ]
#     targets = [DATA_DIR / file for file in file_output]

#     return {
#         "actions": [
#             "ipython ./src/pull_bloomberg.py",
#             "ipython ./src/pull_CRSP_Compustat.py",
#             "ipython ./src/pull_CRSP_stock.py",
#             "ipython ./src/pull_fed_yield_curve.py",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "clean": [],  # Don't clean these files by default.
#     }


def task_summary_stats():
    """ """
    file_dep = ["./src/example_table.py"]
    file_output = [
        "example_table.tex",
        "pandas_to_latex_simple_table1.tex",
    ]
    targets = [OUTPUT_DIR / file for file in file_output]

    return {
        "actions": [
            "ipython ./src/example_table.py",
            "ipython ./src/pandas_to_latex_demo.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


def task_example_plot():
    """Example plots"""
    file_dep = [Path("./src") / file for file in ["example_plot.py", "pull_fred.py"]]
    file_output = ["example_plot.png"]
    targets = [OUTPUT_DIR / file for file in file_output]

    return {
        "actions": [
            # "date 1>&2",
            # "time ipython ./src/example_plot.py",
            "ipython ./src/example_plot.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


def task_chart_repo_rates():
    """Example charts for Chart Book"""
    file_dep = [
        "./src/pull_fred.py",
        "./src/chart_relative_repo_rates.py",
    ]
    targets = [
        DATA_DIR / "repo_public.parquet",
        DATA_DIR / "repo_public.xlsx",
        DATA_DIR / "repo_public_relative_fed.parquet",
        DATA_DIR / "repo_public_relative_fed.xlsx",
        OUTPUT_DIR / "repo_rates.html",
        OUTPUT_DIR / "repo_rates_normalized.html",
        OUTPUT_DIR / "repo_rates_normalized_w_balance_sheet.html",
    ]

    return {
        "actions": [
            # "date 1>&2",
            # "time ipython ./src/chart_relative_repo_rates.py",
            "ipython ./src/chart_relative_repo_rates.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


notebook_tasks = {
    "01_example_notebook_interactive.ipynb": {
        "file_dep": [],
        "targets": [],
    },
    "02_example_with_dependencies.ipynb": {
        "file_dep": ["./src/pull_fred.py"],
        "targets": [Path(OUTPUT_DIR) / "GDP_graph.png"],
    },
    "03_public_repo_summary_charts.ipynb": {
        "file_dep": [
            "./src/pull_fred.py",
            "./src/pull_ofr_api_data.py",
            "./src/pull_public_repo_data.py",
        ],
        "targets": [
            OUTPUT_DIR / "repo_rate_spikes_and_relative_reserves_levels.png",
            OUTPUT_DIR / "rates_relative_to_midpoint.png",
        ],
    },
}


def task_convert_notebooks_to_scripts():
    """Convert notebooks to script form to detect changes to source code rather
    than to the notebook's metadata.
    """
    build_dir = Path(OUTPUT_DIR)
    build_dir.mkdir(parents=True, exist_ok=True)

    for notebook in notebook_tasks.keys():
        notebook_name = notebook.split(".")[0]
        yield {
            "name": notebook,
            "actions": [
                # jupyter_execute_notebook(notebook_name),
                # jupyter_to_html(notebook_name),
                # copy_notebook_to_folder(notebook_name, Path("./src"), "./docs_src/notebooks/"),
                jupyter_clear_output(notebook_name),
                jupyter_to_python(notebook_name, build_dir),
            ],
            "file_dep": [Path("./src") / notebook],
            "targets": [OUTPUT_DIR / f"_{notebook_name}.py"],
            "clean": True,
            "verbosity": 0,
        }


# fmt: off
def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """

    for notebook in notebook_tasks.keys():
        notebook_name = notebook.split(".")[0]
        yield {
            "name": notebook,
            "actions": [
                """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
                jupyter_execute_notebook(notebook_name),
                jupyter_to_html(notebook_name),
                copy_notebook_to_folder(
                    notebook_name, Path("./src"), "./_docs/notebooks/"
                ),
                jupyter_clear_output(notebook_name),
                # jupyter_to_python(notebook_name, build_dir),
                """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
            ],
            "file_dep": [
                OUTPUT_DIR / f"_{notebook_name}.py",
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                BASE_DIR / "_docs" / "notebooks" / f"{notebook_name}.ipynb",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
            # "verbosity": 1,
        }
# fmt: on


# ###############################################################
# ## Task below is for LaTeX compilation
# ###############################################################

def task_compile_latex_docs():
    """Compile the LaTeX documents to PDFs"""
    file_dep = [
        "./reports/report_example.tex",
        "./reports/my_article_header.sty",
        "./reports/slides_example.tex",
        "./reports/my_beamer_header.sty",
        "./reports/my_common_header.sty",
        "./reports/report_simple_example.tex",
        "./reports/slides_simple_example.tex",
        "./src/example_plot.py",
        "./src/example_table.py",
    ]
    targets = [
        "./reports/report_example.pdf",
        "./reports/slides_example.pdf",
        "./reports/report_simple_example.pdf",
        "./reports/slides_simple_example.pdf",
    ]

    return {
        "actions": [
            # My custom LaTeX templates
            "latexmk -xelatex -halt-on-error -cd ./reports/report_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report_example.tex",  # Clean
            "latexmk -xelatex -halt-on-error -cd ./reports/slides_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_example.tex",  # Clean
            # Simple templates based on small adjustments to Overleaf templates
            "latexmk -xelatex -halt-on-error -cd ./reports/report_simple_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report_simple_example.tex",  # Clean
            "latexmk -xelatex -halt-on-error -cd ./reports/slides_simple_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_simple_example.tex",  # Clean
            #
            # Example of compiling and cleaning in another directory. This often fails, so I don't use it
            # f"latexmk -xelatex -halt-on-error -cd -output-directory=../_output/ ./reports/report_example.tex",  # Compile
            # f"latexmk -xelatex -halt-on-error -c -cd -output-directory=../_output/ ./reports/report_example.tex",  # Clean
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }

# ###############################################################
# ## Sphinx documentation
# ###############################################################



pipeline_doc_file_deps = pipeline_publish.get_file_deps(base_dir=BASE_DIR)
generated_md_targets = pipeline_publish.get_targets(base_dir=BASE_DIR)


def task_pipeline_publish():
    """Create Pipeline Docs for Use in Sphinx"""

    file_dep = [
        "./src/pipeline_publish.py",
        "./docs_src/conf.py",
        "./README.md",
        "./pipeline.json",
        "./docs_src/_templates/chart_entry_bottom.md",
        "./docs_src/_templates/chart_entry_top.md",
        "./docs_src/_templates/pipeline_specs.md",
        "./docs_src/_templates/dataframe_specs.md",
        "./docs_src/charts.md",
        "./docs_src/index.md",
        *pipeline_doc_file_deps,
    ]

    targets = [
        *generated_md_targets,
    ]

    return {
        "actions": ["ipython ./src/pipeline_publish.py"],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


notebook_sphinx_pages = [
    "./_docs/_build/html/notebooks/" + notebook.split(".")[0] + ".html"
    for notebook in notebook_tasks.keys()
]
sphinx_targets = [
    "./_docs/_build/html/index.html",
    "./_docs/_build/html/myst_markdown_demos.html",
    "./_docs/_build/html/apidocs/index.html",
    *notebook_sphinx_pages,
]


def task_compile_sphinx_docs():
    """Compile Sphinx Docs"""
    notebook_scripts = [
        OUTPUT_DIR / ("_" + notebook.split(".")[0] + ".py")
        for notebook in notebook_tasks.keys()
    ]
    file_dep = [
        "./docs_src/conf.py",
        "./docs_src/index.md",
        "./docs_src/myst_markdown_demos.md",
        "./docs_src/notebooks.md",
        *notebook_scripts,
        "./README.md",
        "./pipeline.json",
        "./src/pipeline_publish.py",
        "./docs_src/charts.md",
        # Pipeline docs
        "./src/pipeline_publish.py",
        "./docs_src/_templates/chart_entry_bottom.md",
        "./docs_src/_templates/chart_entry_top.md",
        "./docs_src/_templates/pipeline_specs.md",
        "./docs_src/_templates/dataframe_specs.md",
        *pipeline_doc_file_deps,
    ]

    return {
        "actions": [
            "rsync -lr --exclude=charts --exclude=dataframes --exclude=notebooks --exclude=index.md --exclude=pipelines.md --exclude=dataframes.md ./docs_src/ ./_docs/",
            "sphinx-build -M html ./_docs/ ./_docs/_build",
        ],  # Use docs as build destination     
        # "actions": ["sphinx-build -M html ./docs/ ./docs/_build"], # Previous standard organization
        "targets": sphinx_targets,
        "file_dep": file_dep,
        "task_dep": ["run_notebooks", "pipeline_publish"],
        "clean": True,
    }


def copy_build_files_to_docs_publishing_dir(docs_publish_dir=DOCS_PUBLISH_DIR):
    """
    Copy build files to the docs build directory.

    This function copies files and directories from the build directory to the
    docs publishing directory. It iterates over the files and directories in the
    'docs/html' directory and copies them to the corresponding location in the
    'docs_publish_dir'. If a file or directory already exists in the target
    location, it is removed before copying.

    Additionally, this function creates a '.nojekyll' file in the
    'docs_publish_dir' if it doesn't already exist.

    Note that I'm using by default the "docs" directory as the build
    directory. It is also the publishing directory. I just need
    to copy the files out of the HTML sub-directory into the
    root of the publishing directory.
    """
    # shutil.rmtree(docs_publish_dir, ignore_errors=True)
    # shutil.copytree(BUILD_DIR, docs_publish_dir)
    docs_publish_dir = Path(docs_publish_dir)

    for item in (Path("./docs") / "html").iterdir():
        if item.is_file():
            target_file = docs_publish_dir / item.name
            if target_file.exists():
                target_file.unlink()
            shutil.copy2(item, docs_publish_dir)
        elif item.is_dir():
            target_dir = docs_publish_dir / item.name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(item, target_dir)

    nojekyll_file = docs_publish_dir / ".nojekyll"
    if not nojekyll_file.exists():
        nojekyll_file.touch()


# def task_copy_built_docs_to_publishing_dir():
#     """copy_built_docs_to_publishing_dir

#     # For example, convert this:
#     # Copy files from this:
#     ['./docs/html/index.html',
#     './docs/html/myst_markdown_demos.html',
#     './docs/html/apidocs/index.html']

#     # to this:
#     [WindowsPath('docs/index.html'),
#     WindowsPath('docs/myst_markdown_demos.html'),
#     WindowsPath('docs/apidocs/index.html')]
#     """
#     file_dep = sphinx_targets
#     targets = [
#         Path(DOCS_PUBLISH_DIR) / Path(*Path(file).parts[2:]) for file in sphinx_targets
#     ]

#     return {
#         "actions": [
#             copy_build_files_to_docs_publishing_dir,
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "clean": True,
#     }


###############################################################
## Uncomment the task below if you have R installed. See README
###############################################################


# def task_install_r_packages():
#     """Example R plots"""
#     file_dep = [
#         "r_requirements.txt",
#         "./src/install_packages.R",
#     ]
#     targets = [OUTPUT_DIR / "R_packages_installed.txt"]

#     return {
#         "actions": [
#             "Rscript ./src/install_packages.R",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "clean": True,
#     }


# def task_example_r_script():
#     """Example R plots"""
#     file_dep = [
#         "./src/pull_fred.py",
#         "./src/example_r_plot.R"
#     ]
#     targets = [
#         OUTPUT_DIR / "example_r_plot.png",
#     ]

#     return {
#         "actions": [
#             "Rscript ./src/example_r_plot.R",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "task_dep": ["pull_fred"],
#         "clean": True,
#     }


# rmarkdown_tasks = {
#     "04_example_regressions.Rmd": {
#         "file_dep": ["./src/pull_fred.py"],
#         "targets": [],
#     },
#     # "04_example_regressions.Rmd": {
#     #     "file_dep": ["./src/pull_fred.py"],
#     #     "targets": [],
#     # },
# }


# def task_knit_RMarkdown_files():
#     """Preps the RMarkdown files for presentation format.
#     This will knit the RMarkdown files for easier sharing of results.
#     """
#     # def knit_string(file):
#     #     return f"""Rscript -e "library(rmarkdown); rmarkdown::render('./src/04_example_regressions.Rmd', output_format='html_document', output_dir='./_output/')"""
#     str_output_dir = str(OUTPUT_DIR).replace("\\", "/")
#     def knit_string(file):
#         """
#         Properly escapes the quotes and concatenates so that this will run.
#         The single line version above was harder to get right because of weird
#         quotation escaping errors.

#         Example command:
#         Rscript -e "library(rmarkdown); rmarkdown::render('./src/04_example_regressions.Rmd', output_format='html_document', output_dir='./_output/')
#         """
#         return (
#             "Rscript -e "
#             '"library(rmarkdown); '
#             f"rmarkdown::render('./src/{file}.Rmd', "
#             "output_format='html_document', "
#             f"output_dir='{str_output_dir}')\""
#         )

#     for notebook in rmarkdown_tasks.keys():
#         notebook_name = notebook.split(".")[0]
#         file_dep = [f"./src/{notebook}", *rmarkdown_tasks[notebook]["file_dep"]]
#         html_file = f"{notebook_name}.html"
#         targets = [f"{OUTPUT_DIR / html_file}", *rmarkdown_tasks[notebook]["targets"]]
#         actions = [
#             # "module use -a /opt/aws_opt/Modulefiles",
#             # "module load R/4.2.2",
#             knit_string(notebook_name)
#         ]

#         yield {
#             "name": notebook,
#             "actions": actions,
#             "file_dep": file_dep,
#             "targets": targets,
#             "clean": True,
#             # "verbosity": 1,
#         }


###################################################################
## Uncomment the task below if you have Stata installed. See README
###################################################################

# if OS_TYPE == "windows":
#     STATA_COMMAND = f"{config.STATA_EXE} /e"
# elif OS_TYPE == "nix":
#     STATA_COMMAND = f"{config.STATA_EXE} -b"
# else:
#     raise ValueError(f"OS_TYPE {OS_TYPE} is unknown")

# def task_example_stata_script():
#     """Example Stata plots

#     Make sure to run
#     ```
#     net install doenv, from(https://github.com/vikjam/doenv/raw/master/) replace
#     ```
#     first to install the doenv package: https://github.com/vikjam/doenv.
#     """
#     file_dep = [
#         "./src/pull_fred.py",
#         "./src/example_stata_plot.do",
#     ]
#     targets = [
#         OUTPUT_DIR / "example_stata_plot.png",
#     ]
#     return {
#         "actions": [
#             f"{STATA_COMMAND} do ./src/example_stata_plot.do",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "task_dep": ["pull_fred"],
#         "clean": True,
#         "verbosity": 2,
#     }
