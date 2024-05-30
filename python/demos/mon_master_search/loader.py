# cSpell: disable
import fnmatch
import tarfile
from pathlib import Path
from typing import Iterator

import json_repair
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from loguru import logger

from python.ai_core.embeddings import EmbeddingsFactory
from python.ai_core.loaders import load_docs_from_jsonl, save_docs_to_jsonl
from python.ai_core.vector_store import VectorStoreFactory
from python.demos.mon_master_search.model_subset import (
    InformationsPedagogiques,
    ParcoursFormations,
)


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
        "eta_uai": formation.etab.desgn_etab.eta_uai,
        "eta_libelle": formation.etab.desgn_etab.eta_libelle,
        "eta_name": formation.etab.desgn_etab.eta_name,
    }
    for dmn in formation.dnms:
        if dmn.parcours:
            for partour in dmn.parcours:
                metadata_for = {
                    "title": partour.intitule_parcours,
                    "source": f"inmp:{partour.for_inmp}",
                    "modalite_enseignement": str(partour.modalite_enseignement),
                    "licences_conseillees": str(partour.licences_conseillees),
                }
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
            metadata_for = {
                "source": f"inm:{dmn.for_inm}",
                "title": dmn.dom_libelle,
                "modalite_enseignement": str(dmn.modalite_enseignement),
                "licences_conseillees": str(dmn.licences_conseillees),
            }
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


REPO = Path("/mnt/c/Users/a184094/OneDrive - Eviden/_En cours/mon_master/")
FILES = REPO / "synthesis.json"


def load_embeddings():
    #    EMBEDDINGS_MODEL = "multilingual_MiniLM_local"
    EMBEDDINGS_MODEL = "camembert_large_local"

    embeddings_factory = EmbeddingsFactory(embeddings_id=EMBEDDINGS_MODEL)
    vector_factory = VectorStoreFactory(
        id="Chroma",
        embeddings_factory=embeddings_factory,
        collection_name="offres_formation",
        index_document=True,
    )
    docs = list(load_docs_from_jsonl(FILES))

    logger.info(
        f"There are {vector_factory.document_count()} documents in  vector store"
    )

    logger.info(
        f"add {len(docs)} documents to vector store: {vector_factory.description}"
    )
    # vector_factory.get()
    for doc in docs:
        try:
            vector_factory.add_documents([doc])
        except Exception as ex:
            logger.warning(f"cannot add {doc.metadata['source']} - {ex}")


if __name__ == "__main__":
    assert REPO.exists

    if True:
        loader = offre_formation_loader(REPO / "Offres_2024.tgz")
        processed = list(loader.load())
        save_docs_to_jsonl(processed, FILES)

    # load_embeddings()
