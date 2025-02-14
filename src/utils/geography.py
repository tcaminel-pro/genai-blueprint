""" "
Functions to deal with geography areas.

Mostly wrapper around 'constructive_geometries' library

Copyright (C) 2023 Eviden. All rights reserved
"""

from typing import cast

# import country_converter as coco  ##

### WARNING : Country-converter is GPL licensed !
# Consider https://github.com/pycountry/pycountry (LGPL), geonames, ..


def get_country_code_or_area(geo: str) -> str | None:
    """
    Return the ISO2 code if geo is a valid country name or ISO country code, or accepted areas.
    (current version: only "World", "Global", "Glo", "RoW', "Rest of World")
    ex: "GB", "UK", "GBR", "Great Britain" and "United Kingdom" will return "GB"
    Return None if not found
    """
    if geo.lower() in ["world", "glo", "global"]:
        code = "GLO"
    elif geo.lower() in ["rest-of-world", "row", "rest of world", "rest of the world"]:
        code = "RoW"
    else:
        if "-" in geo and len(geo.split("-")[0]) <= 3:  # IN-xx / US-xx / BR-xx / AUS-xx
            geo = geo.split("-")[0]
        code = cast(str, coco.convert(names=geo, to="ISO2"))
    if code == "not found":
        code = None
    return code
