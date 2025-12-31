"""Command-line interface for ALSmuse."""

import click


@click.group()
@click.version_option()
def main():
    """ALSmuse - Analyze Ableton Live sets for music video planning."""
    pass


@main.command()
@click.argument("als_file", type=click.Path(exists=True))
def analyze(als_file):
    """Analyze an Ableton Live Set file."""
    click.echo(f"Analyzing: {als_file}")
    # TODO: Implement analysis


if __name__ == "__main__":
    main()
