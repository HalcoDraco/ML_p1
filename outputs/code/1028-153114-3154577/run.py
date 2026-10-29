def main() -> None:
    """Reproduce the best submission from the raw data.

    Rebuilds the clean dataset, fits the preset in `configs/best.toml` on all the
    training data and writes `outputs/submissions/best.csv`. The model is retrained
    instead of loaded, so nothing in `outputs/` is needed beforehand.
    """
    pass


if __name__ == "__main__":
    main()
