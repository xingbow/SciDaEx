from functools import partial

from langchain.chains.combine_documents import collapse_docs, split_list_of_docs
from langchain_core.prompts import PromptTemplate, format_document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

try:
    import globalVariable as GV
except:
    import app.dataService.globalVariable as GV

def summarize_docs(docs, openai_key = GV.openai_key):
    """
    Summarize a list of documents using a map-reduce approach with language models.

    This function takes a list of Document objects, summarizes each document individually,
    and then combines these summaries into a final coherent summary.

    Args:
        docs (List[Document]): A list of Document objects to be summarized.
        openai_key (str, optional): OpenAI API key for authentication. Defaults to GV.openai_key.

    Returns:
        str: A coherent summary of all input documents.

    Process:
    1. Initialize the language model (LLM) with specific parameters.
    2. Define a chain to summarize individual documents (map step).
    3. Define a chain to collapse multiple summaries (reduce step).
    4. Apply the map-reduce process to generate the final summary.
    """
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0, openai_api_key = openai_key, model_kwargs={"seed": 42})

    # Define prompt and method for converting Document to string
    document_prompt = PromptTemplate.from_template("{page_content}")
    partial_format_document = partial(format_document, prompt=document_prompt)

    # Chain for summarizing individual documents (map step)
    map_chain = (
        {"context": partial_format_document}
        # | PromptTemplate.from_template("Summarize this content:\n\n{context}")
        | PromptTemplate.from_template("Summarize the content of different papers:\n\n{context}")
        | llm
        | StrOutputParser()
    )

    # Wrapper chain to preserve original Document metadata
    map_as_doc_chain = (
        RunnableParallel({"doc": RunnablePassthrough(), "content": map_chain})
        | (lambda x: Document(page_content=x["content"], metadata=x["doc"].metadata))
    ).with_config(run_name="Summarize (return doc)")

    # Helper function to format documents for the collapse chain
    def format_docs(docs):
        return "\n\n".join(partial_format_document(doc) for doc in docs)

    # Chain for collapsing multiple summaries (reduce step)
    collapse_chain = (
        {"context": format_docs}
        | PromptTemplate.from_template("Collapse the content of different papers:\n\n{context}")
        | llm
        | StrOutputParser()
    )

    def get_num_tokens(docs):
        return llm.get_num_tokens(format_docs(docs))

    def collapse(
        docs,
        config,
        token_max=5000,
    ):
        collapse_ct = 1
        while get_num_tokens(docs) > token_max:
            config["run_name"] = f"Collapse {collapse_ct}"
            invoke = partial(collapse_chain.invoke, config=config)
            split_docs = split_list_of_docs(docs, get_num_tokens, token_max)
            docs = [collapse_docs(_docs, invoke) for _docs in split_docs]
            collapse_ct += 1
        return docs

    # Chain for final reduction of collapsed summaries
    reduce_chain = (
        {"context": format_docs}
        | PromptTemplate.from_template("Combine these summaries of different papers to produce a coherent one beginning with 'these papers':\n\n{context}")
        | llm
        | StrOutputParser()
    ).with_config(run_name="Reduce")

    # The final full chain
    map_reduce = (map_as_doc_chain.map() | collapse | reduce_chain).with_config(
        run_name="Map reduce"
    )

    return map_reduce.invoke(docs, config={"max_concurrency": 5})