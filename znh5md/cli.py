import importlib.metadata
import pathlib

import ase.io
import ase.visualize
import tqdm
import typer

app = typer.Typer()


@app.command()
def view(file: str):
    """Use ASE GUI to view a H5MD Trajectory."""
    import znh5md

    typer.echo(f"Loading atoms from {file}")

    data = znh5md.ASEH5MD(file)
    ase.visualize.view(data.get_atoms_list())


@app.command()
def export(file: str, output: str):
    """Export a H5MD File into the output file."""
    import znh5md

    data = znh5md.ASEH5MD(file)
    for atom in tqdm.tqdm(data.get_atoms_list(), ncols=120):
        ase.io.write(output, atom, append=True)


@app.command()
def convert(file: str, db_file: str):
    """Save a trajectory as H5MD File."""
    import znh5md

    db = znh5md.io.DataWriter(db_file)
    if not pathlib.Path(db_file).exists():
        db.initialize_database_groups()

    db.add(znh5md.io.ASEFileReader(file))


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
