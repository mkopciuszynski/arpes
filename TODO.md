# TODO (2023/06/27)

Original author does not seem to maintain this package. Every bug must be fixed by myself...

- Highly maintainable

  - More typing annotation
  - Use ruff for linting and follow the message as possible
  - Follow Zen of Python: "There should be one-- and preferably only one --obvious way to do it. Although that way may not be obvious at first unless you're Dutch."
  - Don't be afraid to remove the function (I'll bet you don't use that one.)

- Follow the recent python (>3.10)

  - User pathlib.Path instead os.path

- Check the functionality

  - ConversionKxKy class
  
  - .band_analysis_utils import param_getter, param_stderr_getter
  - Check type of the argument set at lf.Mmodel: Is it really lf.Model? lf.ModelResult is better?


- rye for packaging
  - tidiy up yaml files
