# Sulfur Simulation

View the `sulfur_simulation` package documentation [here](https://matt-ord.github.io/sulfur_simulation/).

## Setup

This project uses a **Dev Container** (via **VS Code**) and **[`uv`](https://github.com/astral-sh/uv)** for fast, reliable Python dependency management.

### Launch the Dev Container

1. Open the project in VS Code.

2. In VS Code, open the Command Palette (`Ctrl+Shift+P`) and run:

   ```
   Dev Containers: Reopen in Container
   ```

   This will build the container if needed and connect you to the development environment.

### Dependency Management

This project uses **[`uv`](https://github.com/astral-sh/uv)** for dependency management.
To add a package, use the following commands:

```bash
uv add <package-name>
uv add --dev <package-name>
```

This adds the package to `pyproject.toml` and updates the lockfile,
where `--dev` adds it to the development dependencies.

## Features to add:

1. Rewrite update_position to calculate probabilities for lattice to enable particle interaction and other effects

2. Add functionality to simulate square and hcp lattice
