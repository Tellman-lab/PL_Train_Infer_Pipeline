from setuptools import setup, find_packages

setup(
    # this sets the package name that will be installed
    name="PL_Support_Codes",
    version="0.0.1",
    install_requires=[
        "requests", "toml",
        'importlib-metadata; python_version > "3.9"', "hydra-core",
        "opencv-python", "imagecodecs", "yapf", "pytorch_lightning"
    ],
    # this add the __init__.py files in the package directories so that code can call them 
    packages=find_packages(
        # All keyword arguments below are optional:
        where=".",  # '.' by default
        include=["PL_Support_Codes"],  # ['*'] by default
        exclude=["st_water_seg.egg-info", "dist",
                 ".vscode"],  # empty by default
    ),
)
