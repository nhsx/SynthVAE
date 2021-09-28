# Contributing
We love contributions! We've compiled these docs to help you understand our contribution guidelines. If you still have questions, please [contact us](mailto:analytics-unit@nhsx.nhs.uk), we'd be super happy to help.
## Contents of this file

- [Code of conduct](#code-of-conduct)
- [Folder structure](#folder-structure)
- [Commit hygiene](#commit-hygiene)
- [Updating Changelog](#updating-changelog)


## Code of Conduct
Please read [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) before contributing.


## Commit hygiene

Please see the GDS [Git style guide in the 'How to store source code' page of the GDS Way](https://gds-way.cloudapps.digital/standards/source-code.html#commit-messages), which describes how we prefer Git history and commit messages to read.

## Updating the Changelog

If you open a GitHub pull request on this repo, please update `CHANGELOG` to reflect your contribution.

Add your entry under `Unreleased` as: 
- `Breaking changes`
- `New features`
- `Fixes`

Internal changes to the project that are not part of the public API do not need changelog entries, for example fixing the CI build server.

These sections follow [semantic versioning](https://semver.org/spec/v2.0.0.html), where:

- `Breaking changes` corresponds to a `major` (1.X.X) change.
- `New features` corresponds to a `minor` (X.1.X) change.
- `Fixes` corresponds to a `patch` (X.X.1) change.

See the [`CHANGELOG.md`](./CHANGELOG.md) for an example for how this looks.