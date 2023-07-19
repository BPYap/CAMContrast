import os
import setuptools

setuptools.setup(
    name="camcontrast",
    version="0.0",
    packages=["camcontrast"]
)

if not os.path.exists("SupContrast/__init__.py"):
    open("SupContrast/__init__.py", 'a').close()

setuptools.setup(
    name="SupContrast",
    packages=["SupContrast"]
)
