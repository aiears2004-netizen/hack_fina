import time
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# =============================
# CONFIGURATION
# =============================

EMBED = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DIM = 384  # embedding size for the above model
MIN_SIMILARITY = 0.82  # dynamic threshold (best performance)
TOP_K = 5

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

# Create FAISS index using Cosine Similarity
index = faiss.IndexFlatIP(VECTOR_DIM)

QA_CACHE = FAISS(
    embedding_function=EMBED,
    index=index,
    docstore=FAISS._docstore_cls({}),
    index_to_docstore_id={},
)

# =============================
# RETRIEVAL FUNCTION
# =============================


def retrieve_relevant_context(question):
    try:
        results = QA_CACHE.similarity_search_with_score(question, k=TOP_K)
    except Exception:
        return []

    relevant_answers = []

    for doc, score in results:
        if score >= MIN_SIMILARITY:
            ans = doc.metadata.get("answer")
            if ans:
                relevant_answers.append(ans)

    return relevant_answers


# =============================
# BUILD THE FINAL PROMPT
# =============================


def build_prompt(context_list, question):
    base_instructions = """
You are a helpful AI assistant. Use only the provided CONTEXT to answer.
If the answer is not found in CONTEXT, answer normally without hallucinating.

CONTEXT:
{}
""".format(
        "\n\n".join(context_list)
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", base_instructions), ("user", question)]
    )

    return prompt


# =============================
# MAIN FUNCTION
# =============================


def get_answer(user_question):
    # STEP 1: Retrieve similar past Q-A
    context_answers = retrieve_relevant_context(user_question)

    # STEP 2: Build LLM prompt
    prompt = build_prompt(context_answers, user_question)

    # STEP 3: Generate answer
    answer = llm(prompt.format_messages()).content

    # STEP 4: Store Q-A in FAISS memory
    QA_CACHE.add_texts(
        texts=[user_question], metadatas=[{"answer": answer, "ts": time.time()}]
    )

    return answer


# =============================
# EXAMPLE USAGE
# =============================

if _name_ == "_main_":
    while True:
        q = input("\nUser: ")
        ans = get_answer(q)
        print("\nAI:", ans)
