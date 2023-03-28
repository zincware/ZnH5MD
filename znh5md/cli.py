import typer
import tempfile
import importlib.metadata
import subprocess
import ase.io

app = typer.Typer()

@app.command()
def view(file: str):
    """Use ASE GUI to view a H5MD Trajectory."""
    import znh5md

    with tempfile.NamedTemporaryFile(suffix=".xyz") as tmp:
        typer.echo(f"Writing file {file} to temporary file {tmp.name}")
        data = znh5md.ASEH5MD(file)
        ase.io.write(tmp.name, data.get_atoms_list())

@app.command()
def export(file: str, output:str):
    """export a H5MD File into the output file"""
    import znh5md
    data = znh5md.ASEH5MD(file)
    ase.io.write(output, data.get_atoms_list())

def version_callback(value: bool) -> None:
    """Get the installed dask4dvc version."""
    if value:
        typer.echo(f"znh5md {importlib.metadata.version('znh5md')}")
        raise typer.Exit()

@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", callback=version_callback, is_eager=True
    ),
) -> None:
    """Dask4DVC CLI callback.
    Run the DVC graph or DVC experiments in parallel using dask.
    """
    _ = version  # this would be greyed out otherwise