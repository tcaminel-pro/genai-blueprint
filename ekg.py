#!/usr/bin/env python3
"""EKG CLI entry point.

Main command-line interface for the Enhanced Knowledge Graph (EKG) system.
This provides a single 'ekg' command with subcommands for managing opportunity data.

Usage:
    ekg add --key <opportunity_key>    Add opportunity data to KB
    ekg delete                         Delete entire KB
    ekg query                          Interactive Cypher query shell
    ekg info                           Display DB info and schema  
    ekg export-html                    Export HTML visualization

Examples:
    ekg add --key cnes-venus-tma
    ekg query --query "MATCH (n) RETURN count(n)"
    ekg info
    ekg export-html --output-dir ./output
"""

from demos.ekg.cli_commands_ekg import app

if __name__ == "__main__":
    app()