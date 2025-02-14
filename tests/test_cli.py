from typer.testing import CliRunner

from src.main_cli import cli_app, define_other_commands

runner = CliRunner()


def test_app() -> None:
    define_other_commands(cli_app)
    result = runner.invoke(cli_app, ["echo", "Hello world"])
    assert result.exit_code == 0
    assert "Hello world" in result.stdout
