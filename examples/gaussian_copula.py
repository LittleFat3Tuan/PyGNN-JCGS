"""Train a fixed-bandwidth GMMN on Gaussian or absolute-Gaussian copula data."""

from .common import run


def main():
    """Run the Gaussian copula tutorial from the command line."""
    run("gaussian")


if __name__ == "__main__":
    main()
