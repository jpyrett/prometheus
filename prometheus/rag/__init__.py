"""RAG library: DataAgent text files -> LangChain Documents -> Chroma -> search.

    loaders.py    news/<SYMBOL>.txt -> Documents          (Part 3)
    store.py      the embedding model and the Chroma index (Part 4)
    ingest.py     chunk, encode, store                    (Part 4)
    retriever.py  search by meaning, with filters         (Part 5)

It reads the files the DataAgent writes (news/<SYMBOL>.txt, later data/filings and data/press_releases)
and never imports DataAgent: the file format in docs/DataAgent.md is the contract between them.

Step-by-step build: labs/module3_rag/hpw_to_do_langchain_rag.md
Design decisions:   docs/design_choices.md (entries 10 to 20)
"""
