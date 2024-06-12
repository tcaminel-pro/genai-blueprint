"""
retrieve relevant data from Informatica XML file
"""

from pathlib import Path
from typing import Tuple

import graphviz
import xmltodict
from loguru import logger

INDENT = 4 * " "


def get_value_by_name(name: str, table_att: dict) -> str | None:
    return next((item["@VALUE"] for item in table_att if item["@NAME"] == name), None)


def extract_from_xml(input_file: Path):
    with open(input_file, "r") as f:
        xml = f.read()
    dxml = xmltodict.parse(xml)  # TODO: Maybe set force_list to True ?

    for powermart_value in dxml.values():
        repository = powermart_value.get("REPOSITORY")
        if repository:
            folder = repository.get("FOLDER")
            if folder:
                result = extract_from_folder(folder)
                non_empty_lines = "\n".join(
                    [line for line in result.split("\n") if line.strip()]
                )
                with open(input_file.with_suffix(".txt"), "w") as f:
                    xml = f.write(non_empty_lines)


def extract_from_folder(folder: dict) -> Tuple[str, set]:
    result = ""
    # SOURCE
    sources = folder.get("SOURCE")
    if sources:
        result += f"{0*INDENT}Sources:\n"
        for source in sources:
            result += f"{1*INDENT}Table: {source.get('@NAME')}, database = {source.get('@DATABASETYPE')}"
            field_info = [
                f"{f.get('@NAME')} ({f.get('@DATATYPE')})"
                for f in source.get("SOURCEFIELD")
            ]
            field_list_str = ", ".join(field_info)
            result += f"{2*INDENT}Fields: {field_list_str})\n"
    # TARGET
    targets = folder.get("TARGET")
    if targets:
        result += f"{0*INDENT}Targets:\n"
        for target in targets:
            result += f"{1*INDENT}Table: {target.get('@NAME')}, database: {target.get('@DATABASETYPE')}\n"
            field_info = [
                f"{f.get('@NAME')}" for f in target.get("TARGETFIELD")
            ]  # {f.get('@DATATYPE')}
            field_list_str = ", ".join(field_info)
            result += f"{2*INDENT}Fields: {field_list_str})\n"
    # MAPPING
    mapping = folder.get("MAPPING")
    if mapping:
        result += f"{1*INDENT}Mapping: {mapping.get('@NAME')}\n"
        # TRANSFORMATION
        for transfo in mapping.get("TRANSFORMATION"):
            type = transfo.get("@TYPE")
            field_info = [
                f"{f.get('@NAME')}" for f in transfo.get("TRANSFORMFIELD")
            ]  # {f.get('@DATATYPE')}
            field_list_str = ", ".join(field_info)
            result += f"{2*INDENT}Transformation: {transfo.get('@NAME')} ({type}) \n"
            if type == "Lookup Procedure":
                table_attr = transfo.get("TABLEATTRIBUTE")
                v = get_value_by_name("Lookup condition", table_attr)
                result += f"{3*INDENT}Lookup condition: {v}\n"
            elif type == "Filter":
                table_attr = transfo.get("TABLEATTRIBUTE")
                v = get_value_by_name("Filter Condition", table_attr)
                result += f"{3*INDENT}Filter Condition: {v}\n"
            elif type == "Expression":
                fields = transfo.get("TRANSFORMFIELD")
                for field in fields:
                    if field.get("@NAME") != field.get("@EXPRESSION"):
                        if field.get("@EXPRESSION"):
                            # TODO : Use Regexp
                            exp = field.get("@EXPRESSION").split("--")[0]
                            result += f"{3*INDENT}{field.get('@NAME')} <= {exp}\n "
                        pass
            elif type == "Sorter":
                pass
            elif type == "Aggregator":
                fields = transfo.get("TRANSFORMFIELD")
                for field in fields:
                    exp_type = field.get("@EXPRESSIONTYPE")
                    if exp_type == "GENERAL":
                        if field.get("@NAME") != field.get("@EXPRESSION"):
                            result += f"{3*INDENT}{field.get('@NAME')} <= {field.get('@EXPRESSION')} \n"
                    elif exp_type == "GROUPBY":
                        result += f"{3*INDENT}GROUP_BY: {field.get('@NAME')}\n"
                    else:
                        pass  # TODO; Check
            else:
                logger.warning(f"Unknown Transformation type: {type}")
        # CONNECTOR
        result += f"{1*INDENT}Connectors:\n"
        conn_set = set()
        for f in mapping.get("CONNECTOR"):
            conn_set.add((f.get("@FROMINSTANCE"), f.get("@TOINSTANCE")))
        for from_inst, to_inst in conn_set:
            info = f"{from_inst} ==> {to_inst}"
            result += f"{2*INDENT}{info}\n"
    return result, conn_set


def create_graph_dot(base_file: Path, conn_set: set):
    dot = graphviz.Digraph(comment="informatica 1", graph_attr={"rankdir": "LR"})
    for from_inst, to_inst in conn_set:
        dot.node(from_inst)
        dot.node(to_inst)
        dot.edge(from_inst, to_inst)
        dot.render("diagram.gv").replace("\\", "/")
