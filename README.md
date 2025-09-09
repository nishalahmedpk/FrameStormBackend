22# FrameStormBackend

FrameStormBackend is a FastAPI-based backend for generating videos, blogs, and images. It supports agentic editing workflows and provides APIs for seamless interaction with a frontend application.

## Project Structure

- **api.py**  
  FastAPI application that exposes endpoints for the frontend. It calls workflows from `videoworkflow.py` for video and blog generation, while the image generation API is called directly.

- **videoworkflow.py**  
  Contains the main workflows for video generation, agentic video editing, and blog post generation.

- **videogeneration.py**  
  Implements the logic to generate and download videos, used by both the video generation and agentic editing workflows.

## Setup Instructions

   ```bash
   git clone <your-repo-url>
   cd FrameStormBackend
   pip install uv
   uv run fastapi run api.py
```

## Available Endpoints

Video Generation & Agentic Editing: Interact via api.py which internally calls videoworkflow.py.

Blog Generation: Available through the same workflow in videoworkflow.py.

Image Generation: Direct API calls handled in api.py.


## Notes

Ensure that all dependencies for video and image generation are installed.

This backend is designed to integrate easily with a frontend application for real-time content creation.

Still in MVP Phase