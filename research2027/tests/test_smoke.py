from mockingbird2027 import __version__
from mockingbird2027.cli import main


def test_package_and_cli_smoke(capsys):
    assert __version__ == "0.1.0"
    assert main(["smoke"]) == 0
    assert "Research2027 scaffold OK" in capsys.readouterr().out
