import importlib.metadata

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

    io = znh5md.IO(file)
    ase.visualize.view(io[:])


@app.command()
def export(file: str, output: str):
    """Export a H5MD File into the output file."""
    import znh5md

    data = znh5md.IO(file)
    for atom in tqdm.tqdm(data[:], ncols=120):
        ase.io.write(output, atom, append=True)


@app.command()
def convert(file: str, db_file: str):
    """Save a trajectory as H5MD File."""
    import znh5md

    atoms = ase.io.read(file, ":")
    znh5md.write(db_file, atoms)


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
    """ZnH5MD: A H5MD file viewer and converter."""
    _ = version  # this would be greyed out otherwise
