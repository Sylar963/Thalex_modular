from setuptools import setup, find_packages

setup(
    name="thalex_modular",
    version="0.1.0",
    # Find packages in root (src) and in thalex_py
    packages=find_packages(where="src") + find_packages(where="thalex_py"),
    package_dir={
        "": "src",
        "thalex": "thalex_py/thalex",
    },
    install_requires=[
        "fastapi",
        "uvicorn",
        "aiohttp",
        "websockets",
        "orjson",
        "uvloop",
        "numpy",
        "pandas",
        "asyncpg",
        "eth_account",
        "python-dotenv",
        "pydantic",
        "pydantic-settings",
        "requests",  # Added from thalex_py requirements
    ],
)
