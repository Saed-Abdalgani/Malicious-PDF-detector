from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="malicious-pdf-detector",
    version="1.0.0",
    author="Saed Abdalgani",
    author_email="example@example.com",
    description="A fail-closed malicious PDF detector with calibrated static analysis and abstention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Saed-Abdalgani/Malicious-PDF-detector",
    packages=find_packages(include=['src', 'src.*', 'app', 'app.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "malpdf-remediate=src.run_all:main",
            "malpdf-app=app.streamlit_app:main",
        ],
    },
)
