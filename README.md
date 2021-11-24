# CoopIHC

Link to the project site: https://jgori-ouistiti.github.io/CoopIHC

# Dependencies

CoopIHC uses pipenv to manage dependencies. You should run 

pipenv install

to get the required dependencies.

CoopIHC does not ship a requirements.txt file, but you can recreate it from the Pipfile with

pipenv lock -r > requirements.txt

if you want to use that mechanism.


Some examples require matplotlib, which itself requires a graphical backend to display graphs. Since many backends exist, CoopIHC does not require a specific one as a dependency. If the examples do not display, then you should likely either install a graphical backend (e.g. pyqt5, see matplotlib documentation for others) with pipenv, or, if you have a system-wide install for some backend (likely) go to the configuration file of your pipfile and allow system-wide packages to be installed.
