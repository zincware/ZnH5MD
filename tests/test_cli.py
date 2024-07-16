"""Test the CLI module."""

import ase.build
import ase.io
from typer.testing import CliRunner

import znh5md
from znh5md.cli import app  # Import the Typer app

runner = CliRunner()


def test_export(tmp_path):
    tmp_input = tmp_path / "input.h5md"
    tmp_output = tmp_path / "output.xyz"

    atoms = ase.build.molecule("H2O")
    znh5md.write(tmp_input, atoms)

    assert tmp_input.exists()

    result = runner.invoke(app, ["export", tmp_input.as_posix(), tmp_output.as_posix()])
    assert result.exit_code == 0
    assert tmp_output.exists()

    new_atoms = ase.io.read(tmp_output)
    assert new_atoms.get_chemical_symbols() == atoms.get_chemical_symbols()


def test_convert(tmp_path):
    tmp_input = tmp_path / "input.xyz"
    tmp_output = tmp_path / "output.h5md"

    atoms = ase.build.molecule("H2O")
    ase.io.write(tmp_input, atoms)

    assert tmp_input.exists()

    result = runner.invoke(
        app, ["convert", tmp_input.as_posix(), tmp_output.as_posix()]
    )
    assert result.exit_code == 0
    converted_atoms = znh5md.IO(tmp_output)[:]
    assert len(converted_atoms) > 0


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "znh5md" in result.stdout
