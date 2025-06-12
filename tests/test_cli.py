from typer.testing import CliRunner

from src.main.cli import cli_app

runner = CliRunner()


def test_app() -> None:
    result = runner.invoke(cli_app, ["echo", "Hello world"])
    assert result.exit_code == 0
    assert "Hello world" in result.stdout
