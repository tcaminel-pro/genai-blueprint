# cSpell: disable
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class JsonModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class DesgnEtab(JsonModel):
    eta_uai: str
    eta_libelle: str
    eta_name: str


class Etab(JsonModel):
    desgn_etab: DesgnEtab


class LieuxItem(JsonModel):
    site: str
    ville: str
    geo: str


class InformationsPedagogiques(JsonModel):
    lien_fiche: str
    mot_cle_disciplinaire: Optional[List[str]] = None
    mot_cle_sectoriel: Optional[List[str]] = None
    mot_cle_metier: Optional[List[str]] = None
    mot_cle_libre: Optional[List[str]] = None


class Parcour(JsonModel):
    for_inmp: str
    intitule_parcours: str
    informations_pedagogiques: InformationsPedagogiques | None = None
    modalite_enseignement: List[str] | None = None


class Dnm(JsonModel):
    for_inm: str | None = None
    for_intitule: str
    dom_libelle: List[str]
    informations_pedagogiques: Optional[InformationsPedagogiques] = None
    parcours: Optional[List[Parcour]] = None


class ParcoursFormations(JsonModel):
    etab: Etab
    dnms: List[Dnm]
