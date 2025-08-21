import gradio as gr
from ollama_engine import ollama_setup
from agents.workflow import WorkflowGraph
from config import (LLM_MODEL_NAME, 
                    EMBEDDING_MODEL_NAME,
                    CHROMA_DB_PATH,
                    VECTOR_SEARCH_K,
                    HYBRID_RETRIEVER_WEIGHTS)


def setup_ollama_on_load():
    try:
        ollama_setup(LLM_MODEL_NAME)
        ollama_setup(EMBEDDING_MODEL_NAME)
        gr.Info("✅ Ollama setup complete!")
    except Exception as e:
        gr.Warning(f"⚠️ Ollama setup error: {e}")

def main():
    wf = WorkflowGraph(
    llm_model_name=LLM_MODEL_NAME,
    embedding_model_name=EMBEDDING_MODEL_NAME,
    chroma_dir=CHROMA_DB_PATH,
    vector_k=VECTOR_SEARCH_K,
    retriever_weights=HYBRID_RETRIEVER_WEIGHTS,
    )
    
    with gr.Blocks(title="Document Processor") as demo:
        gr.Markdown("# 📄 Multi-Document Processor")

        demo.load()

        # --- Create 5 rows of inputs ---
        row_dropdowns = []
        row_textboxes = []
        row_filepickers = []

        for i in range(5):
            with gr.Row():
                dd = gr.Dropdown(
                    ["Link", "PDF"],
                    value="Link",
                    label=f"Row {i+1}: Select Type",
                    visible=(i == 0),
                    scale=1,
                )
                tb = gr.Textbox(
                    label=f"Document URL {i+1}",
                    visible=(i == 0),
                    scale=3,
                )
                fp = gr.File(
                    label=f"Upload PDF {i+1}",
                    file_types=[".pdf"],
                    type="filepath",
                    visible=False,
                    scale=3,
                )

            row_dropdowns.append(dd)
            row_textboxes.append(tb)
            row_filepickers.append(fp)

            # toggle textbox/file depending on dropdown
            def toggle_input(choice, tb=row_textboxes[i], fp=row_filepickers[i]):
                if choice == "Link":
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)

            dd.change(
                fn=toggle_input,
                inputs=dd,
                outputs=[tb, fp],
            )

        # --- Control buttons ---
        with gr.Row():
            add_btn = gr.Button("➕ Add Document", scale=1)
            query_input = gr.Textbox(label="Enter Your Query", placeholder="What is the main topic of the documents? Or ask a question about a topic to search the web for related papers.")    
            submit_btn = gr.Button("Submit Documents", variant="primary", scale=2)

        output_message = gr.Textbox(label="Processing Log", interactive=False)

        # --- State to track how many rows are visible ---
        counter = gr.State(1)

        def add_input(shown_count):
            """Reveal only the NEXT hidden row without resetting previous ones"""
            if shown_count >= 5:
                return [gr.update()] * (5 * 3) + [shown_count]  # do nothing if already max
            next_index = shown_count  # 0-based
            updates = []
            for i in range(5):
                if i == next_index:
                    # reveal this row's dropdown + textbox (link default), keep file hidden
                    updates.append(gr.update(visible=True))
                    updates.append(gr.update(visible=True))
                    updates.append(gr.update(visible=False))
                else:
                    # no change for all other rows
                    updates.append(gr.update())
                    updates.append(gr.update())
                    updates.append(gr.update())
            return updates + [shown_count + 1]

        add_btn.click(
            fn=add_input,
            inputs=counter,
            outputs=[
                comp
                for triple in zip(row_dropdowns, row_textboxes, row_filepickers)
                for comp in triple
            ] + [counter],
        )

        # --- Handle submission ---
        def handle_submission(*all_inputs):
            print("Processing documents...")
            files = []
            query = all_inputs[-1]
            for i in range(5):
                choice = all_inputs[i]
                text_val = all_inputs[5 + i]
                file_val = all_inputs[10 + i]
                if choice == "Link" and text_val:
                    files.append({"type": "url", "value": text_val})
                elif choice == "PDF" and file_val:
                    files.append({"type": "pdf", "value": file_val})
            if not files:
                print("No valid inputs provided.")
                return "⚠️ Please provide at least one valid input."

            state = wf.run(question=query, files=files)
            # display concise log
            rel = state.get("relevance")
            if rel == "CAN_ANSWER":
                return f"✅ Answered from uploaded docs.\n\n{state.get('final_answer','')}"
            return f"🌐 Used web search (Reflexion). Relevance={rel}\n\n{state.get('final_answer','')}"

        submit_btn.click(
            fn=handle_submission,
            inputs=row_dropdowns + row_textboxes + row_filepickers + [query_input],
            outputs=output_message,
        )
    
    return demo


if __name__ == "__main__":
    setup_ollama_on_load()  # Ensure Ollama is set up before launching
    demo = main()
    demo.launch(server_name="0.0.0.0", server_port=7860)
