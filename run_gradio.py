#!/usr/bin/env python
"""Start the standalone Week3 Gradio client."""

from frontend.app import CSS, build_demo


if __name__ == "__main__":
    build_demo().queue().launch(
        server_name="127.0.0.1", server_port=7860, share=False, css=CSS
    )
