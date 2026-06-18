


from __future__ import annotations

import sys

import least_squares_prediction as lsp


def main() -> None:
    sys.argv = [
        "least_squares_prediction.py",
        "--max-nfev",
        "4000",
        "--max-points",
        "0",
        "--starts",
        "8",
        "--merge-existing",
    ]
    args = lsp.parse_args()
    lsp.run(args)


if __name__ == "__main__":
    main()
