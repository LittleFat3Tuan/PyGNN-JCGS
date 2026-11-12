"""Train an adaptive Gaussian-copula GMMN with bandwidth and validation logs."""

from .common import run


def main():
    """Run the adaptive bandwidth tutorial from the command line."""
    run("gaussian", adaptive=True)


if __name__ == "__main__":
    main()
