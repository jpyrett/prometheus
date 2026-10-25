"""RAG library: DataAgent text files -> LangChain Documents -> Chroma -> retriever.

It reads the files the DataAgent writes (news/<SYMBOL>.txt, later data/filings and data/press_releases)
and never imports DataAgent: the file format in docs/DataAgent.md is the contract between them.

Step-by-step build: labs/module3_rag/hpw_to_do_langchain_rag.md
Design decisions:   docs/design_choices.md (entries 10 to 14)
"""
