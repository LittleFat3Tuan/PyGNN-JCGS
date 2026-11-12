"""Train a fixed-bandwidth GMMN on synthetic Clayton copula observations."""

from .common import run


def main():
    """Run the Clayton copula tutorial from the command line."""
    run("clayton")


if __name__ == "__main__":
    main()
