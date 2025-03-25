from setuptools import setup, find_packages

setup(
    name="thalex_py",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'websockets>=10.0',
        'plotly>=5.3.0',
        'python-jwt>=4.0.0',
        'cryptography>=3.4.0',
        'python-dotenv>=0.19.0',
        'asyncio>=3.4.3',
        'typing>=3.7.4.3'
    ]
) 