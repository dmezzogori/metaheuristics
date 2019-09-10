from setuptools import setup

setup(
    name="metaheuristics",
    version="1.0",
    description="Metaheuristics Algorithms",
    author="Davide Mezzogori",
    author_email="davide@mezzogori.com",
    packages=["metaheuristics"],
    zip_safe=False,
    install_requires=["colorama", "numpy", "progressbar2"],
)
