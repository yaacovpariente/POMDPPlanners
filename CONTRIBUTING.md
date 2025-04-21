# Contributing Guide

Thank you for considering contributing to this project! Whether you're fixing a bug, adding a new feature, improving documentation, or suggesting an idea, your input is appreciated.

This guide will walk you through how to contribute effectively and respectfully.

---

## 🧭 Introduction

This project aims to standardize simulation studies for research and provide reliable implementations of planning algorithms for industrial applications. Contributions of all kinds are welcome — from code and documentation to bug reports and ideas.

This guide is for anyone interested in contributing, whether you're a beginner or experienced open-source contributor.

---

## 🔧 How to Contribute

We welcome a variety of contributions:

- 🐛 Reporting bugs  
- 💡 Requesting features  
- ✍️ Improving documentation  
- 🧪 Writing or improving tests  
- 💻 Submitting code (bug fixes, new features, refactoring)

Before contributing a major change, please open an issue or discussion to ensure the change is aligned with the project's direction.

---

## 💻 Development Setup

### Prerequisites

- Python 3.8 or higher
- Required tools:
  - pip (Python package installer)
  - pytest (for running tests)
  - black (for code formatting)
  - pylint (for static code analysis)
  - make (optional, for running checks)

### Steps

1. Fork the repository  
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/project-name.git
   cd project-name
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Set up pre-commit hooks (this will automatically format your code before each commit):
   ```bash
   pre-commit install
   ```
5. Run tests and linting:
   ```bash
   pytest
   black .
   flake8 .
   pylint POMDPPlanners/
   ```

---

## 🧪 Coding Guidelines

- Follow the project's code style (e.g., PEP8 for Python, clang-format for C++)  
- Use linters and formatters consistently (e.g., `black`, `flake8`, `clang-format`)  
- Write clear, concise, and descriptive commit messages  
- Keep changes focused: avoid mixing unrelated fixes or features in a single pull request  
- Include tests for new features or bug fixes  
- Update or add documentation as needed

---

## 🧬 Making a Contribution

1. Create a new branch:
   ```bash
   git checkout -b <issue label>/<issue number>
   ```
   For example, if you take an issue with label "feature" and the issue number is 14, then the branch should be named feature/14.

2. Make your changes locally

3. Run tests and linters:
   ```bash
   pytest
   black .
   pylint POMDPPlanners/
   ```
   Note that `black` is automatically run on commit through pre-commit hooks, but you can run it manually as shown above.

4. Commit your changes:
   ```bash
   git add .
   git commit -m "Descriptive commit message"
   ```

5. Push to your fork:
   ```bash
   git push origin <issue label>/<issue number>
   ```

6. Open a pull request against the `master` branch. Include:
   - A clear title and description of your change  
   - A reference to any related issue numbers, if applicable  
   - Any relevant screenshots, benchmarks, or documentation links

7. After PR is closed - delete the development working branch ```<issue label>/<issue number>```.
---

## 🧾 Review Process

- Your pull request will be reviewed for style, clarity, correctness, and alignment with the project goals  
- Reviews may suggest changes or ask for clarification  
- Automated checks (CI, linters, tests) must pass before merging  
- You may be asked to squash or rebase commits  
- Response time is typically within a few days

---

## 🔒 Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md)

We are committed to a welcoming and harassment-free experience for everyone. Be respectful, inclusive, and constructive.

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project.

See the [LICENSE](LICENSE.md) file for more details.

If a Contributor License Agreement (CLA) is required, it will be documented here.

---

## 🛠️ Tools and Automation

This project uses the following tools:

- **CI/CD**: All pull requests run through automated pipelines for testing, formatting, and linting  
- **Formatter**: `black` (Python), `clang-format` (C++)  
- **Linter**: `pylint`  
- **Tests**: `pytest`
- **Docs**: [e.g., Sphinx, MkDocs]

---

## 💡 Tips for New Contributors

- Look for [good first issues](https://github.com/yaacovpariente/POMDPPlanners/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22Good%20first%20issue%22) label.
- Don't hesitate to ask questions — we love helping
