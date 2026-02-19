import logging
import warnings
from collections.abc import Generator
from typing import Any
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langchain_huggingface import HuggingFaceEmbeddings
except (ImportError, Exception):
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 150
MMR_K            = 4
MMR_FETCH_K      = 20
LLM_MODEL        = "gemini-1.5-flash"
LLM_TEMPERATURE  = 0.2

SYSTEM_PROMPT = (
    "You are a precise document analysis assistant. "
    "Answer questions using ONLY the information found in the provided document context. "
    "If the answer is not present in the context, say so clearly — do not speculate. "
    "Keep answers concise and cite relevant details from the document when possible."
)

# Prompt used to rewrite user queries into better retrieval queries
QUERY_REWRITE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are an expert at reformulating questions to maximise document retrieval quality.\n"
        "Rewrite the following question into a clear, specific search query that will retrieve "
        "the most relevant passages from a document. Return ONLY the rewritten query, nothing else.\n\n"
        "Original question: {question}\n"
        "Rewritten query:"
    ),
)


def _format_sources(source_docs: list[Any]) -> list[dict]:

    sources = []
    for doc in source_docs:
        snippet = doc.page_content.strip()
        # Truncate long snippets for display
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        score = doc.metadata.get("relevance_score", None)
        sources.append({"snippet": snippet, "relevance_score": score})
    return sources


class RAGAgent:

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.vector_store: FAISS | None = None
        self.conversation_chain: ConversationalRetrievalChain | None = None

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        logger.info("Initialising LLM: %s", LLM_MODEL)
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=self.api_key,
            temperature=LLM_TEMPERATURE,
            system_instruction=SYSTEM_PROMPT,
        )

        # Separate low-temperature LLM for query rewriting
        # (deterministic — we want a consistent rewrite, not a creative one)
        self.rewrite_llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=self.api_key,
            temperature=0.0,
        )

        self._reset_memory()


    def process_document(self, text: str) -> bool:
 
        if not text or not text.strip():
            logger.warning("process_document called with empty text.")
            return False

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = splitter.split_text(text)

        if not chunks:
            logger.warning("Text splitting produced 0 chunks.")
            return False

        logger.info("Embedding %d chunks...", len(chunks))
        self.vector_store = FAISS.from_texts(texts=chunks, embedding=self.embeddings)
        self.conversation_chain = self._build_chain()
        logger.info("Document indexed and RAG chain ready.")
        return True

    def get_response(self, query: str) -> tuple[str, list[dict]]:

        if not self.vector_store:
            return "No document has been processed yet. Please upload a file first.", []

        if not self.conversation_chain:
            self.conversation_chain = self._build_chain()

        try:
            # Step 1 — Query rewriting
            rewritten = self._rewrite_query(query)
            logger.info("Query rewritten: '%s' → '%s'", query, rewritten)

            # Step 2 — Retrieval + generation
            response = self.conversation_chain.invoke({"question": rewritten})
            answer   = response.get("answer", "No answer returned.")

            # Step 3 — Source attribution
            source_docs = response.get("source_documents", [])
            sources     = _format_sources(source_docs)

            return answer, sources

        except Exception as exc:
            logger.exception("Error generating response: %s", exc)
            return f"Error generating response: {exc}", []

    def get_response_stream(self, query: str) -> Generator[str, None, None]:

        if not self.vector_store:
            yield "No document has been processed yet."
            return

        try:
            rewritten = self._rewrite_query(query)
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": MMR_K, "fetch_k": MMR_FETCH_K},
            )
            # Retrieve relevant chunks
            docs = retriever.invoke(rewritten)
            context = "\n\n".join(d.page_content for d in docs)

            # Build a single-turn prompt with context for streaming
            prompt = (
                f"Using the following document context, answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )

            # Stream tokens
            for chunk in self.llm.stream(prompt):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    yield token

        except Exception as exc:
            logger.exception("Streaming error: %s", exc)
            yield f"Error: {exc}"

    def reset(self) -> None:
        """Clear the document index and conversation history."""
        logger.info("Resetting RAG agent state.")
        self.vector_store = None
        self.conversation_chain = None
        self._reset_memory()

    def _rewrite_query(self, query: str) -> str:

        try:
            prompt  = QUERY_REWRITE_PROMPT.format(question=query)
            result  = self.rewrite_llm.invoke(prompt)
            rewritten = result.content.strip() if hasattr(result, "content") else str(result).strip()
            return rewritten if rewritten else query
        except Exception as exc:
            logger.warning("Query rewrite failed, using original: %s", exc)
            return query

    def _reset_memory(self) -> None:
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

    def _build_chain(self) -> ConversationalRetrievalChain:
        """Build a ConversationalRetrievalChain with MMR retrieval."""
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": MMR_K, "fetch_k": MMR_FETCH_K},
        )
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,   # Required for source attribution
            verbose=False,
        )