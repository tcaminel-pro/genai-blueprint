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
    licences_conseillees: Optional[List[str]] = None
    modalite_enseignement: List[str] | None = None


class Dnm(JsonModel):
    for_inm: str | None = None
    for_intitule: str
    dom_libelle: List[str]
    informations_pedagogiques: Optional[InformationsPedagogiques] = None
    parcours: Optional[List[Parcour]] = None
    licences_conseillees: Optional[List[str]] = None
    modalite_enseignement: List[str] | None = None


class ParcoursFormations(JsonModel):
    etab: Etab
    dnms: List[Dnm]


LICENCES_CONSEILLEES = [
    "Administration économique et sociale",
    "Administration et échanges internationaux",
    "Administration publique",
    "Arts",
    "Arts du spectacle",
    "Arts plastiques",
    "Chimie",
    "Cinéma",
    "Droit",
    "Droit canonique",
    "Droit français - Droits étrangers",
    "Droit, histoire de l'art",
    "Economie",
    "Economie et gestion",
    "Economie, science politique",
    "Electronique, énergie électrique, automatique",
    "Environnements océaniens",
    "Etudes culturelles",
    "Etudes Européennes et Internationales",
    "Frontières du vivant",
    "Génie civil",
    "Géographie et aménagement",
    "Gestion",
    "Histoire",
    "Histoire de l'art et archéologie",
    "Humanités",
    "Information-communication",
    "Informatique",
    "Langues étrangères appliquées",
    "Langues, littératures et civilisations étrangères et régionales",
    "Lettres",
    "Lettres, langues",
    "Licence intégrée franco-allemande en droit",
    "Mathématiques",
    "Mathématiques et informatique appliquées aux sciences humaines et sociales",
    "Mécanique",
    "Métiers de l'enseignement du premier degré",
    "Musicologie",
    "Philosophie",
    "Physique",
    "Physique, chimie",
    "Psychologie",
    "Science politique",
    "Sciences de la Terre",
    "Sciences de la vie",
    "Sciences de la vie et de la Terre",
    "Sciences de l'éducation",
    "Sciences de l'Homme, anthropologie, ethnologie",
    "Sciences du langage",
    "Sciences et Humanités",
    "Sciences et techniques des activités physiques et sportives",
    "Sciences et techniques des activités physiques et sportives : activité physique adaptée et santé",
    "Sciences et techniques des activités physiques et sportives : éducation et motricité",
    "Sciences et techniques des activités physiques et sportives : entraînement sportif",
    "Sciences et techniques des activités physiques et sportives : ergonomie du sport et performance motrice",
    "Sciences et techniques des activités physiques et sportives : management du sport",
    "Sciences et technologies",
    "Sciences pour la santé",
    "Sciences pour l'ingénieur",
    "Sciences sanitaires et sociales",
    "Sciences sociales",
    "Sociologie",
    "Théologie",
    "Théologie catholique",
    "Théologie protestante",
    "Tourisme",
    "Toutes licences",
]

MODELITE_ENSEIGNEMENT = [
    "Formation initiale",
    "Formation continue",
    "Formation à distance",
    "Alternance - Contrat de professionnalisation",
    "Alternance - Apprentissage",
    "Hybride",
]
