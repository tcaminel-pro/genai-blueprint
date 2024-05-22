# cSpell: disable
import fnmatch
import json
import tarfile
from pathlib import Path
from typing import Iterator

import json_repair
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from loguru import logger

from python.ai_core.embeddings import EmbeddingsFactory
from python.ai_core.vector_store import vector_store_factory
from python.demos.mon_master_search.model_subset import (
    InformationsPedagogiques,
    ParcoursFormations,
)

# cSpell:disable


def format_info_pedago(intitule: str, info_pedago: InformationsPedagogiques):
    content = []
    content.append(f"intitulé: {intitule}")
    if info := info_pedago.mot_cle_disciplinaire:
        content.append(f"disciplines: {','.join(info)}")
    if info := info_pedago.mot_cle_sectoriel:
        content.append(f"secteurs: {','.join(info)}")
    if info := info_pedago.mot_cle_metier:
        content.append(f"métier: {','.join(info)}")
    if info := info_pedago.mot_cle_libre:
        content.append(f"autre: {','.join(info)}")
    return "\n".join(content)


def process_json(source: str, formation: ParcoursFormations) -> Iterator[Document]:
    logger.debug(f"load {source}")

    metadata_offre = {
        "source": source,
        "eta_uai": formation.etab.desgn_etab.eta_uai,
        "eta_libelle": formation.etab.desgn_etab.eta_libelle,
        "eta_name": formation.etab.desgn_etab.eta_name,
    }
    for dmn in formation.dnms:
        if dmn.parcours:
            for partour in dmn.parcours:
                metadata_for = {"inmp": partour.for_inmp}
                if partour.informations_pedagogiques:
                    content = format_info_pedago(
                        partour.intitule_parcours,
                        partour.informations_pedagogiques,
                    )
                    if lien := partour.informations_pedagogiques.lien_fiche:
                        metadata_for |= {"lien_fiche": lien}

                    yield Document(
                        page_content=content,
                        metadata=metadata_for | metadata_offre,
                    )

        dmn_info_pedago = dmn.informations_pedagogiques
        if dmn_info_pedago:
            content = format_info_pedago("".join(dmn.dom_libelle), dmn_info_pedago)
            metadata_for = {"inm": dmn.for_inm}
            if lien := dmn_info_pedago.lien_fiche:
                metadata_for |= {"lien_fiche": lien}
            yield Document(
                page_content=content,
                metadata=metadata_for | metadata_offre,
            )


class offre_formation_loader(BaseLoader):
    def __init__(self, doc_list: Path):
        self.parcours_json_archive = doc_list

    def lazy_load(self) -> Iterator[Document]:
        with tarfile.open(self.parcours_json_archive, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile() and fnmatch.fnmatch(member.name, "Offre_*.json"):
                    raise Exception("incorrect file in archive: {member}")
                file_obj = tar.extractfile(member)
                assert file_obj

                json_file = file_obj.read().decode("utf-8")
                json_obj = json_repair.loads(json_file)
                parcours = ParcoursFormations(**json_obj)  # type: ignore

                for doc in process_json(str(member.name), parcours):
                    yield doc


def load_embeddings():
    EMBEDDINGS_MODEL = "camembert_large_local"

    embeddings_factory = EmbeddingsFactory(embeddings_id=EMBEDDINGS_MODEL)
    vector_factory = vector_store_factory(
        id="Chroma",
        embeddings_factory=embeddings_factory,
        collection_name="offres_formation",
    )


if __name__ == "__main__":
    REPO = Path("/mnt/c/Users/a184094/OneDrive - Eviden/_En cours/mon_master/")
    assert REPO.exists

    loader = offre_formation_loader(REPO / "Offres_2024.tgz")
    processed = list(loader.load())

    json_data = json.dumps([item.dict() for item in processed], indent=4)
    with open(REPO / "synthesis.json", "w") as io:
        io.write(json_data)
