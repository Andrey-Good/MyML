from setuptools import setup, find_packages

setup(
    name="my_tools",  # Как будешь импортировать
    version="0.1.0",
    packages=find_packages(),
    install_requires=[  # Что нужно, чтобы оно работало
        "numpy",
        "pandas",
        "torch",
        "matplotlib",
        # 'catboost' # можно добавить, если часто юзаешь
    ],
    author="AndreyG0oD",
    description="My collection of ML crutches and bicycles",
)
