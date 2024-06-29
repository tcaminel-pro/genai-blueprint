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
    for_inm: str
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


ACRONYMS = {
    "AMO": "Assistance à Maîtrise d'Ouvrage",
    "ACQ": "ANALYSE-CONTRÔLE-QUALITÉ",
    "APAS": "Activités Physiques Adaptées et Santé",
    "APS": "Activités Physiques et Sportives",
    "APSA": "Activités Physiques, Sportives et Artistiques",
    "BAPT": "Biotechnologie et Amélioration des Plantes Tropicales",
    "BME": "Biomedical engineering",
    "BTP": "Bâtiment et Travaux Publics",
    "CAFEP": "Certificat d'Aptitude aux Fonctions d'Enseignement dans les établissements Privés",
    "CAFERUIS": "Certificat d'Aptitude aux Fonctions d'Encadrement et de Responsable d'Unité d'Intervention Sociale",
    "CAPEPS": "Certificat d'aptitude au professorat d'éducation physique et sportive",
    "CAPESA": "Certificat d'Aptitude au Professorat de l'Enseignement du Second degré Agricole",
    "CAPLP": "Certificat d'aptitude aux fonctions d'enseignement du privé",
    "CAPPEI": "Certificat d'Aptitude Professionnelle aux Pratiques de l'Éducation Inclusive",
    "CCA": "Comptabilité, contrôle, audit",
    "CDMAT": "CHIMIE DURABLE-MATÉRIAUX",
    "CDORG": "CHIMIE DURABLE ORGANIQUE",
    "CDPI": "Conception et Développement de Produits Industriels",
    "CGAO": "Contrôle de Gestion et Audit Organisationnel",
    "CGPI": "Conseiller en Gestion de Patrimoine Indépendant",
    "CMT": "Chimie moléculaire et thérapeutique",
    "CPE": "Conseiller principal d'éducation",
    "CTM": "Chimie Théorique Modélisation",
    "DCG": "Diplôme de Comptabilité et de Gestion",
    "DIFLES": "Diplôme Initial de Français Langue Étrangère ou Seconde",
    "DJCE": "Diplôme de Juriste Conseil d'Entreprise",
    "DNL": "Discipline Non Linguistique",
    "DSCG": "Diplôme Supérieur de Comptabilité et de Gestion",
    "DSEM": "des mathématiques",
    "DSP": "Droit et Sciences Politiques",
    "DUMI": "Diplôme Universitaire de Musicien Intervenant",
    "EAD": "Enseignement À Distance",
    "ECODEVA": "Economie du Développement Agricole de l'Environnement, et Alimentation",
    "EGEN": "et Gestion des Écosystèmes Naturels",
    "EMCC": "European Master In Classical Cultures",
    "EPHE": "École Pratique des Hautes Études",
    "EPS": "Edication Physique et Sportive",
    "EPUB": "Electronic Publication",
    "ESG": "Environnement, Social et Gouvernance",
    "ESGT": "École Supérieure des Géomètres et Topographes",
    "ESN": "Entreprise de Services du Numérique",
    "FLE": "Français Langue Étrangère",
    "FLES": "Français Langue Étrangère et Seconde",
    "FLI": "Français Langue d'Intégration",
    "FLS": "Français Langue Seconde",
    "FOS": "Français sur Objectif Spécifique",
    "GLM": "Gestion des littoraux et des mers",
    "GPF": "GÉNIE DES PRODUITS FORMULÉS",
    "GPI": "Génie des Procédés Industriels",
    "GRH": "Gestion des ressources humaines",
    "HSE": "Hydrogéologie, Sol et Environnement",
    "IAE": "Institut d'Administration des Entreprises",
    "ICOA": "Ingénierie et eco-COnception des Aliments",
    "IDEC": "Infirmier Diplômé d'État Coordinateur",
    "IDL": "Ingénierie du développement logiciel",
    "IFRS": "International Financial Reporting Standards",
    "IGF": "Ingénierie financière",
    "IJTM": "Institut de Journalisme Tous Médias",
    "IMHE": "Interactions Microorganismes-Hôtes-Environnements",
    "INSPE": "Institut National Supérieur du Professorat et de l'Éducation",
    "IPAG": "Institut de Préparation à l'Administration Générale",
    "IPM": "Interactions Plantes Microorganismes",
    "ISR": "Investissement Socialement Responsable",
    "ISTC": "Institut des Stratégies et Techniques de Communication",
    "LEA": "Langues étrangères appliquée",
    "LEDILANGT": "Lexiques, Discours, Langues et Théories",
    "LLCER": "Langues, littératures et civilisations étrangères et régionales",
    "LSF": "Langue des Signes Française",
    "MCCI": "Médiation Culturelle et Communication Internationale",
    "MEEF": "Métiers de l'Enseignement, de l'Éducation et de la Formation",
    "MESD": "Membrane Engineering for Sustainable Development",
    "MIAGE": "Méthodes Informatiques Appliquées à la Gestion des Entreprises",
    "MIASHS": "Mathématiques, Informatique Appliquées aux Sciences Humaines et Sociales",
    "MOBA": "Management Omnical Banque et Assurance",
    "MOMA": "Moyen-Orient et Maghreb",
    "MPT": "Management Public Territorial",
    "MUTI": "Management des universités et technologies de l'information",
    "NSI": "Numérique et Sciences Informatiques",
    "PCS": "Psychologie clinique de la santé",
    "QHSE": "Qualité, Hygiène, Sécurité, Environnement",
    "QSE": "Qualité Sécurité Environnement",
    "RADMEP": "RADiation and its effects on MicroEelectronics and Phonics Technologies",
    "RSE": "Responsabilité sociale de l'entreprise",
    "RSO": "Responsabilité Sociétale des Organisations",
    "SHS": "Sciences Humaines et Sociales",
    "SHSP": "Sciences Humaines, Sociales et Philosophie",
    "SIAD": "Statistiques pour l'Information et l'Aide à la Décision",
    "SIEF": "Système d’Information Economique et Financière",
    "SII": "sciences industrielles de l'ingénieur",
    "SNT": "Sciences Numériques et Technologie",
    "SOCIALESSCIENCES": "Sciences Sociales",
    "STI": "sciences et techniques industrielles",
    "STM": "SCIENCES ET TECHNOLOGIES DE METAVERS",
    "STMG": "Sciences et Technologies du Management et de la Gestion",
    "STMS": "Sciences et techniques médico-sociales",
    "SVT": "Sciences de la Vie et de la Terre",
    "TAL": "Traitement Automatique des Langues",
    "UX": "Expérience Utilisateur",
    "VFX": "Visual Effects",
    "VNP": "Valorisation des nouveaux patrimoines",
}


STOP_WORDS = [
    "master",
    "mastère",
    "formation",
    "diplome",
    "parcours",
    "intitulé",
    "libelés",
    "license",
    "enseignement",
    "discipline",
    "secteur",
    "metier",
]


EXAMPLE_QUERIES = [
    "Linguistique informatique",
    "Phys nucléaire fusion",
    "IA architecture perf",
    "LLCER",
    "allemant",
    "Droit civil",
    "Biologie cellulaire avancée",
    "Lettres modernes",
    "Droit constitutionnel",
    "Politique etrangere chine",
    "Recherche thermodynamique et applications",
    "Energies renouvel. et économie",
    "Gestion de l'entreprise droit du travail international",
    "Espagnol",
    "Italienne",
    "Ecologie",
    "Ecology",
    "Environnement",
    "Plastique",
    "Veille techno",
    "Professeur des écoles",
    "Langues étrangères",
    "Management",
    "FLE",
    "FLE FLS",
    "FLE FLES",
    "LEA",
    "PCS",
]
