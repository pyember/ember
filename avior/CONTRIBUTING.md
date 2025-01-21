
# Contributing to Avior

Thank you for your interest in contributing! All kinds of contributions are welcome, small or large, including (not limited to) the following. 

- Identifying and reporting any issues or bugs.
- Commenting and assisting others with GitHub Issues.
- Suggesting or implementing new features.
- Improving documentation or contributing how-to guides. 
- Sharing new examples or use-cases.
<!-- - Spreading the word -->

## License

Please see [LICENSE](LICENSE).

## Install

Install the extra requirements for development and testing as follows.
```
pip install -e .[dev]
```

## Testing
Run tests from directory `avior` as follows.
```
python -m unittest discover ./src/avior
```
See [Python unittest docs](https://docs.python.org/3/library/unittest.html) for additional options.


## Contribution Guidelines

### Issues

Please [search existing issues](https://github.com/foundrytechnologies/avior/issues/) before reporting a bug or requesting a feature. If none adequately captures the gap you identified, please [post a new issue](https://github.com/foundrytechnologies/avior/issues/new) and link any related GitHub Issues you discovered.

Alternatively, if you would like to contribute a feature but have some questions or an idea sketch you would like help with, consider starting with a Draft PR! 

### Pull Requests & Code Reviews

To get started, checkout the branch you are building on (usually `main`), create a new branch e.g. `awesome-features`, and make your changes. Before creating a PR with the target branch (e.g. `main`), make sure you have the latest changes to `main`, and are able to merge `main` into `awesome-features`. Other than testing, you are ready to create a PR!

Please use the [PR template](.github/pull_request_template.md) to describe your contribution and expedite review.

### Thank You

Finally, thank you for your support and taking the time to read and follow these guidelines!