import streamlit as st
from vector_generation_module import VectorSearch, ONNXGenerator, build_few_shot_prompt

@st.cache_resource()
def load_vector_search():
    corpus = [
        "All employees must wear safety helmets in the production area.",
        "Visitors should sign in at reception and wear badges.",
        "Fire exits must be kept clear at all times.",
        "PPE such as gloves and goggles are mandatory when handling chemicals."
    ]
    return VectorSearch(corpus)

@st.cache_resource()
def load_generator():
    return ONNXGenerator("distilbart.onnx")

def main():
    st.title("Interactive RAG System with Advanced Prompting & Fallbacks")

    user_query = st.text_input("Ask a question or type here:")

    if user_query:
        vector_search = load_vector_search()
        generator = load_generator()

        retrieved_docs = vector_search.search(user_query, top_k=3)

        max_score = max(score for _, score in retrieved_docs)
        if max_score < 0.3:
            st.warning("I'm sorry, I don't have information on that topic.")
            return

        st.subheader("Top Retrieved Documents")
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            st.write(f"**{i}.** {doc} (Score: {score:.4f})")

        prompt = build_few_shot_prompt(retrieved_docs, user_query)
        answer = generator.generate(prompt)

        # Optional: clean repetitive output heuristic
        if any(kw in answer.lower() for kw in ["employees", "section", "document"]):
            parts = answer.split('.')
            if len(parts) > 1:
                answer = parts[-2].strip() + '.'

        st.subheader("Generated Answer")
        st.write(answer)

if __name__ == "__main__":
    main()
