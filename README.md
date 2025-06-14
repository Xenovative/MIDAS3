# MIDAS3

MIDAS3 is a modular, extensible AI chat and workflow platform designed for advanced LLM interactions, RAG (Retrieval-Augmented Generation), and custom bot management. It integrates multiple AI services, supports document retrieval, and provides a modern, user-friendly web interface.

## Features

- **Chat with LLMs:** Interact with large language models using a clean web UI.
- **Custom Bot Management:** Create, edit, and manage AI bots with custom system prompts, knowledge files, and generation parameters.
- **Retrieval-Augmented Generation (RAG):** Upload documents and enable context-aware Q&A using vector search (ChromaDB).
- **Multiple Model Support:** Easily switch between system models and custom bots.
- **ComfyUI Integration:** Launch and use ComfyUI alongside the main platform.
- **User Preferences:** Customize appearance, model defaults, and more.
- **Real-time Updates:** Conversation titles, bot names, and UI elements update instantly.

## Quick Start

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Xenovative/MIDAS3.git
   cd MIDAS3
   ```

2. **Install dependencies:**
   - Ensure you have [Miniconda/Anaconda](https://docs.conda.io/en/latest/) installed.
   - Run the batch script:
     ```sh
     start_platform.bat
     ```
   - This will:
     - Set up the Conda environment
     - Start Ollama (LLM service)
     - Launch the Flask server
     - Start ComfyUI in a separate console

3. **Access the app:**
   - Open your browser and go to: [http://localhost:5000](http://localhost:5000)

## Requirements

- Windows 10/11
- Miniconda or Anaconda
- Python 3.10+ (managed by Conda)
- [Ollama](https://ollama.com/) (for LLM serving)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (optional, for image workflows)

## Folder Structure

- `start_platform.bat` — Main startup script.
- `app.py` — Flask backend.
- `static/` — Frontend assets (JS, CSS).
- `workflows/` — Example or custom workflow files.
- `docs/` — Upload your documents for RAG here.

## Custom Bots & RAG

- Use the **Manage Bots** button in the sidebar to create custom bots.
- Upload knowledge files to enable retrieval-augmented responses.
- Adjust generation parameters and system prompts per bot.

## License

MIT License
