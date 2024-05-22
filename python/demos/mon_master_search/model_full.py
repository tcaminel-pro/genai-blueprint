# cSpell: disable
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DesgnEtab(BaseModel):
    eta_uai: str
    eta_sigle: str
    eta_libelle: str
    eta_name: str


class Adresse(BaseModel):
    eta_ville: str


class InformationsPedagogiques(BaseModel):
    lien_fiche: str
    mot_cle_disciplinaire: List[str]
    mot_cle_sectoriel: List[str]
    mot_cle_metier: List[str]
    mot_cle_libre: List[str]
    langues: List[str]
    lieux: List[LieuxItem]


class DatesRecrutement(BaseModel):
    date_ouverture_campagne: str
    date_fermeture_campagne: str


class Parcour(BaseModel):
    for_inmp: str
    intitule_parcours: str
    informations_pedagogiques: InformationsPedagogiques
    semestre: int
    dates_recrutement: DatesRecrutement
    licences_conseillees: List[str]
    attendus: List[str]
    criteres: List[str]
    taux_acces: Optional[int] = None
    criteres_examen: List[str]
    formation_ouverte: bool
    modalite_enseignement: Optional[List[str]]


class InformationsPedagogiques1(BaseModel):
    lien_fiche: str
    langues: List[str]
    lieux: List


class DatesRecrutement1(BaseModel):
    date_ouverture_campagne: str
    date_fermeture_campagne: str


class Dnm(BaseModel):
    annee_universitaire: str = Field(..., alias="annee-universitaire")
    for_inm: str
    nature_form: str
    for_intitule: str
    dom_libelle: List[str]
    jury_rectoral: bool
    rncp: str
    frais_scolarite_annuel: FraisScolariteAnnuel
    amenagements: List
    formation_ouverte: bool
    modalite_enseignement: List[str]
    taux_insertion_professionnelle: Dict[str, Any]
    semestre: int
    cal: int
    parcours: List[Parcour]
    informations_pedagogiques: Optional[InformationsPedagogiques1] = None
    dates_recrutement: Optional[DatesRecrutement1] = None
    licences_conseillees: Optional[List[str]] = None
    attendus: Optional[List[str]] = None
    criteres: Optional[List[str]] = None
    taux_acces: Optional[int] = None
    criteres_examen: Optional[List[str]] = None


class Model(BaseModel):
    etab: Optional[Etab] = None
    dnms: Optional[List[Dnm]] = None
