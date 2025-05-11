# Contributing to MakarSpace

We love your input! We want to make contributing to MakarSpace as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

### Contribution Guidelines

#### For Anomaly Detection Models

- Include benchmarks comparing your model to existing ones
- Document the mathematical foundation of your approach
- Validate on both synthetic and real-world datasets (if available)
- Include performance metrics for edge devices (e.g., NVIDIA Jetson)

#### For Synthetic Data Generation

- Document the physics principles behind your simulator
- Include statistical validation of generated data
- Provide examples of edge cases your generator can simulate

#### For Visualization and Explainability

- Focus on intuitive explanations for non-AI experts
- Follow accessibility best practices
- Include examples with real spacecraft telemetry scenarios

## Code of Conduct

### Our Pledge

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone, regardless of age, body
size, visible or invisible disability, ethnicity, sex characteristics, gender
identity and expression, level of experience, education, socio-economic status,
nationality, personal appearance, race, religion, or sexual identity
and orientation.

### Our Standards

Examples of behavior that contributes to a positive environment include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

## License

By contributing, you agree that your contributions will be licensed under the project's Apache 2.0 License.

## Getting Started

Ready to contribute? Here's how to set up MakarSpace for local development.

1. Fork the MakarSpace repo on GitHub.
2. Clone your fork locally:
```
git clone https://github.com/your-username/MakarSpace.git
```
3. Install your local copy into a virtualenv:
```
cd MakarSpace
python -m venv venv
source venv/bin/activate
pip install -e .
```
4. Create a branch for local development:
```
git checkout -b name-of-your-bugfix-or-feature
```
5. Make your changes locally.
6. When you're done making changes, check that your changes pass the tests:
```
pytest
```
7. Commit your changes and push your branch to GitHub:
```
git add .
git commit -m "Your detailed description of your changes."
git push origin name-of-your-bugfix-or-feature
```
8. Submit a pull request through the GitHub website.

Thank you for your contributions!
