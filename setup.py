from distutils.core import setup

setup(
    name = 'metaheuristics',
    packages = ['metaheuristics'],
    version = '0.2',
    license='gpl-3.0',
    description = 'A collection of Metaheuristics Algorithms',
    author = 'Davide Mezzogori',
    author_email="d.mezzogori@me.com",
    url = 'https://github.com/dmezzogori/metaheuristics',
    keywords = ['metaheuristics',],
    install_requires=["colorama", "numpy", "progressbar2"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.7',
    ],
)