# arpes.exceptions module

**exception arpes.exceptions.AnalysisError**

> Bases: `Exception`
> 
> Base class to indicate that something scientific went wrong and was
> not handled in an appropriate way.
> 
> \# Examples
> 
> 1\. A bad fit from scipy.optimize in an internal function or analysis
> routine that could not be handled by the user

**exception arpes.exceptions.AnalysisWarning**

> Bases: `UserWarning`
> 
> Non-fatal, but the user probably forgot something really significant
> that invalidates any science that follows

**exception arpes.exceptions.ConfigurationError**

> Bases: `Exception`
> 
> Indicates that the user needs to supply more configuration. This could
> be setting some directories in which to place plots, or indicating
> what the current workspace being used (i.e. where to look for datasets
> and data) is.

**exception arpes.exceptions.DataPreparationError**

> Bases:
> 
> Indicates that the user needs to perform some data preparation step
> before the analysis can proceed.