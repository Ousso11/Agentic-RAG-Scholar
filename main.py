import gradio as gr
from ollama_engine import ollama_setup
from agents.workflow import WorkflowGraph
from config import (
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    CHROMA_DB_PATH,
    VECTOR_SEARCH_K,
    HYBRID_RETRIEVER_WEIGHTS,
)

from logger  import Logger
logger = Logger(__name__).get_logger()

def setup_ollama_on_load():
    """Try to warm up Ollama models. UI notices are best-effort."""
    try:
        ollama_setup(LLM_MODEL_NAME)
        ollama_setup(EMBEDDING_MODEL_NAME)
    except Exception as e:
        # Gradio notifications created outside a Blocks sometimes don't render.
        # We'll surface any errors in the app log area at runtime instead.
        print(f"Ollama setup error: {e}")


def main():
    MAX_ROWS = 7

    wf = WorkflowGraph(
        llm_model_name=LLM_MODEL_NAME,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        chroma_dir=CHROMA_DB_PATH,
        vector_k=VECTOR_SEARCH_K,
        retriever_weights=HYBRID_RETRIEVER_WEIGHTS,
        logger=logger,
    )

    with gr.Blocks(title="Scholar RAG Agent") as demo:
        gr.Markdown("# 📄 Agentic-RAG Scholar")

        # --- Top horizontal query bar ---
        with gr.Row():
            query_input = gr.Textbox(
                label="Ask your question",
                placeholder="e.g. What are the latest papers on human evaluation in NLP?",
                scale=4,
            )
            with gr.Row():
                add_btn = gr.Button("➕ Add Document", scale=1)
                submit_btn = gr.Button("🔍 Search / Answer", variant="primary", scale=1)
                

        # --- Documents section (rows hidden by default) ---
        row_containers = []  # the Row containers (so we can hide/show the whole row)
        row_dropdowns = []   # each row's Link/PDF choice
        row_textboxes = []   # URL input
        row_filepickers = [] # PDF file input
        row_delete_btns = [] # per-row delete buttons

        def _toggle_input(choice):
            """Switch visibility of URL vs PDF input for a single row."""
            if choice == "Link":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        for i in range(MAX_ROWS):
            with gr.Row(visible=False) as row:
                dd = gr.Dropdown(
                    ["Link", "PDF"],
                    value="Link",
                    label=f"Document {i+1} Type",
                    scale=1,
                )
                tb = gr.Textbox(label=f"Document URL {i+1}", scale=3, visible=True)
                fp = gr.File(
                    label=f"Upload PDF {i+1}",
                    file_types=[".pdf"],
                    type="filepath",
                    visible=False,
                    scale=3,
                )
                del_btn = gr.Button("🗑️ Remove", variant="secondary", scale=0)

                # Toggle URL/PDF inputs when dropdown changes
                dd.change(fn=_toggle_input, inputs=dd, outputs=[tb, fp])

            row_containers.append(row)
            row_dropdowns.append(dd)
            row_textboxes.append(tb)
            row_filepickers.append(fp)
            row_delete_btns.append(del_btn)

        # Area to surface logs/answers
        output_message = gr.Textbox(label="Processing Log", interactive=False)

        # --- State to track which rows are visible ---
        # A list[bool] of length MAX_ROWS
        visible_state = gr.State([False] * MAX_ROWS)

        # Utility to compute updates that reveal the first hidden row
        def add_row(vis_list):
            vis = vis_list.copy()
            try:
                idx = vis.index(False)
            except ValueError:
                # All rows already visible
                updates = [gr.update() for _ in row_containers]
                return updates + [vis]
            vis[idx] = True
            updates = []
            for i in range(MAX_ROWS):
                updates.append(gr.update(visible=vis[i]))  # update each Row container
            return updates + [vis]

        add_btn.click(
            fn=add_row,
            inputs=visible_state,
            outputs=row_containers + [visible_state],
        )

        # Per-row delete (hide) handler factory
        def make_delete_fn(row_index):
            def _delete_row(vis_list):
                vis = vis_list.copy()
                vis[row_index] = False
                row_updates = [gr.update(visible=vis[i]) for i in range(MAX_ROWS)]
                # Also clear that row's inputs when it's hidden
                clear_dd = gr.update(value="Link")
                clear_tb = gr.update(value="")
                clear_fp = gr.update(value=None)
                return row_updates + [vis, clear_dd, clear_tb, clear_fp]
            return _delete_row

        # Wire each delete button
        for i in range(MAX_ROWS):
            row_delete_btns[i].click(
                fn=make_delete_fn(i),
                inputs=visible_state,
                outputs=row_containers + [visible_state, row_dropdowns[i], row_textboxes[i], row_filepickers[i]],
            )

        # --- Handle submission ---
        def handle_submission(*all_inputs):
            files, query = [], all_inputs[-1]
            # Gather inputs row-wise: dropdowns, textboxes, filepickers
            for i in range(MAX_ROWS):
                choice, text_val, file_val = all_inputs[i], all_inputs[MAX_ROWS + i], all_inputs[2 * MAX_ROWS + i]
                # Only consider meaningful entries (empty strings/None are ignored)
                if choice == "Link" and text_val:
                    files.append({"type": "url", "value": text_val})
                elif choice == "PDF" and file_val:
                    files.append({"type": "pdf", "value": file_val})

            try:
                state = wf.run(question=query, files=files)
                rel = state.get("relevance")
                if rel == "CAN_ANSWER":
                    return f"✅ Answered from uploaded docs.\n\n{state.get('final_answer','')}"
                return f"🌐 Used web search (Reflexion). Relevance={rel}\n\n{state.get('final_answer','')}"
            except Exception as e:
                return f"⚠️ Error while processing: {e}"

        submit_btn.click(
            fn=handle_submission,
            inputs=row_dropdowns + row_textboxes + row_filepickers + [query_input],
            outputs=output_message,
        )

    return demo


if __name__ == "__main__":
    setup_ollama_on_load()  # Warm up models before launching
    demo = main()
    demo.launch(server_name="0.0.0.0", server_port=7860)
