"""Check for working pyarpes....

Actually, this check is not so useful...
"""

from importlib.util import find_spec

__all__ = ["check"]


def check() -> None:
    """Verifies certain aspects of the installation and provides guidance broken installations."""

    def verify_qt_tool() -> str | None:
        pip_command = "pip install pyqtgraph"
        warning = (
            "Using qt_tool, the PyARPES version of Image Tool, requires "
            f"pyqtgraph and PySide6:\n\n\tYou can install with: {pip_command}"
        )
        if find_spec("pyqtgraph") is None:
            return warning
        return None

    def verify_igor_pro() -> str | None:
        pip_command = "pip install https://github.com/arafune/igorpy"
        warning = f"For Igor support, install igorpy with: {pip_command}"
        warning_incompatible = (
            "PyARPES requires a patched copy of igorpy, "
            "available at \n\t"
            "https://github.com/arafune/igorpy: "
            f"{pip_command}"
        )
        try:
            import igor

        except ValueError:
            return warning_incompatible
        except (ImportError, AttributeError):
            return warning

        if igor.__version__ <= "0.3":
            raise ValueError

        return None

    checks = [
        ("Igor Pro Support", verify_igor_pro),
        ("qt_tool Support", verify_qt_tool),
    ]

    from colorama import Fore, Style

    print("Checking...")
    for check_name, check_fn in checks:
        initial_str = f"[ ] {check_name}"
        print(initial_str, end="", flush=True)

        failure_message = check_fn()

        print("\b" * len(initial_str) + " " * len(initial_str) + "\b" * len(initial_str), end="")

        if failure_message is None:
            print(f"{Fore.GREEN}[✔] {check_name}{Style.RESET_ALL}")
        else:
            print(
                f"{Fore.RED}[✘] {check_name}: \n\t{failure_message}{Style.RESET_ALL}",
            )
