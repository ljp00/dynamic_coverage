from setuptools import setup, find_packages

setup(
    name="dynamic_coverage",
    version="0.1.0",
    author="Lv Jianpeng",
    description="多无人机动态覆盖控制仿真框架",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pyyaml",
        "scikit-learn",
    ],
)
