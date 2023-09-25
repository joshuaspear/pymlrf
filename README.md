# temp readme

This is a github python repo template. Follow these instructions on how to use this repo as a template for others:
https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/creating-a-repository-from-a-template


## Setting up the repo
* Alter the name of src/python-template to src/<name_of_package>
* Set the "name" parameter of setuptools.setup in setup.py to <name_of_package>

### Logging structure within src/python-template/__init__.py
* A custom logger is defined which adds additional details to logging statements and colours different levels e.g, 'critical' is set to red
  * This has been directly compied from https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
* the logging_name variable can either be hardcoded or set to __name__ depending on the intended usecase. (Disclaimer - I don't know too much about logging!)
* The variable env_name should be set if logging to an output file is required. The value assigned to env_name should align to an environment variable defining the location of the output logging file

## Additional notes
* The guthub actions defined in .github are not properly setup so won't run - they are supposed to be examples.
* For small datafiles used within the package (e.g., small model weights etc), suggest using pkg_resources to load the data. For larger datafiles, suggest pulling from an external repo
