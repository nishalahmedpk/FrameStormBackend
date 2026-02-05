# FrameStorm

An automated video generation pipeline that takes text descriptions and produces complete short-form videos with voiceover, using LLMs, text-to-speech, and text-to-video models. Includes an agent-based chat interface for editing the generated videos.

## Features

- Generates voiceover scripts and visual descriptions from text prompts using Qwen LLM
- Synthesizes speech audio using ElevenLabs TTS
- Creates video clips using Alibaba Dashscope Wan2.1 text-to-video model
- Automatically syncs and combines video clips with audio
- Chat-based editing interface with specialized agents for audio/video/clip modifications
- Optional blog post generation from prompts
- Redis state storage for persistent project data across sessions
- LangGraph-based workflow orchestration with conditional routing
- FastAPI REST endpoints
- Docker Compose setup for deployment

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI REST API                         │
├─────────────────────────────────────────────────────────────┤
│  /generate_video  │  /chat_with_video  │  /generate_blog   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflows                       │
├─────────────────────────────────────────────────────────────┤
│  Video Generation Graph:                                     │
│  Script → Audio → Metadata → Video → Assembly               │
│                                                              │
│  Chat Agent Graph:                                           │
│  Audio Agent → Video Agent → Clip Rearrangement Agent       │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  ┌──────────┐      ┌─────────────┐     ┌─────────────┐
  │ Qwen LLM │      │ ElevenLabs  │     │  Dashscope  │
  │  Engine  │      │     TTS     │     │   Video AI  │
  └──────────┘      └─────────────┘     └─────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Redis State   │
                    │   Storage     │
                    └───────────────┘
```

## Getting Started

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- Redis running locally
- API Keys:
  - [Alibaba Model Studio](https://modelstudio.alibabacloud.com/) (for video/text generation)
  - [ElevenLabs](https://elevenlabs.io/) (for text-to-speech)
  - [LangSmith](https://smith.langchain.com/) (optional, for monitoring)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FrameStormBackend
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Install dependencies with uv**
   ```bash
   uv sync
   ```

4. **Start Redis** (in another terminal)
   ```bash
   redis-server
   ```

5. **Run the API server**
   ```bash
   uv run fastapi run api.py
   ```

The API will be available at `http://localhost:8000` and the interactive documentation at `http://localhost:8000/docs`.

## API Usage

### 1. Generate a Video

```bash
curl -X POST "http://localhost:8000/generate_video" \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "my_viral_video",
    "description": "Create an inspiring 15-second video about technological innovation",
    "generate_video": true
  }'
```

**Parameters:**
- `project_name`: Unique identifier for the project (creates a directory)
- `description`: Natural language description of the desired video
- `generate_video`: Boolean flag to enable/disable actual video generation

**Output:**
- Generated voiceover audio file
- Multiple 5-second video clips
- Final assembled video with synchronized audio
- Project state saved in Redis

### 2. Chat with Video (Editing Agent)

```bash
curl -X POST "http://localhost:8000/chat_with_video" \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "my_viral_video",
    "question": "Make the first clip 3 seconds instead of 5"
  }'
```

**Supported Commands:**
- Rearrange clip order
- Trim clip durations
- Add new video clips
- Remove specific clips
- Regenerate audio with new voiceover
- Generate additional video content

### 3. Generate Blog Content

```bash
curl -X POST "http://localhost:8000/generate_blog" \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "my_viral_video",
    "prompt": "Write about the future of AI in video production"
  }'
```

### 4. Generate Standalone Image

```bash
curl -X POST "http://localhost:8000/generate_image" \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "my_viral_video",
    "prompt": "A futuristic AI laboratory with holographic displays"
  }'
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Backend Framework** | FastAPI |
| **Workflow Orchestration** | LangGraph |
| **LLM Provider** | Qwen (Alibaba Cloud) |
| **Text-to-Speech** | ElevenLabs API |
| **Video Generation** | Alibaba Dashscope Wan2.1 |
| **Video Processing** | MoviePy |
| **State Management** | Redis |
| **Package Manager** | uv |

## System Design

### LangGraph Workflow Engine
The video generation pipeline is implemented as a stateful directed graph with the following nodes:
1. **ScriptGenerator**: Generates voiceover and visual script using Qwen LLM
2. **AudioGenerator**: Synthesizes speech using ElevenLabs API
3. **VideoMetadataGenerator**: Calculates scene timing and generates prompts
4. **VideoGenerator**: Creates video clips via Alibaba Dashscope
5. **VideoAssembler**: Merges clips with audio using MoviePy

### Multi-Agent Chat System
Three specialized agents handle video editing requests:
- **AudioGenerationAgent**: Decides when to regenerate voiceover
- **VideoGenerationAgent**: Determines need for new video clips
- **ClipRearrangementAgent**: Handles trimming, reordering, and deletion

Each agent uses tool-calling capabilities to execute modifications while maintaining conversation context.

### State Persistence
Redis stores project state as JSON, enabling:
- Multi-session editing workflows
- Project recovery after restarts
- Concurrent project management

## Project Structure

```
FrameStormBackend/
├── api.py                  # FastAPI endpoints
├── videoworkflow.py        # LangGraph workflows & agents
├── videogeneration.py      # Alibaba Dashscope integration
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Multi-service orchestration
├── .env.example            # Environment template
└── README.md               # Documentation
```

## Use Cases

- Prototyping short-form video content quickly
- Generating marketing videos from text descriptions
- Converting educational content into video format
- Bulk content creation for social media
- Product demo videos

## Security Notes

- Keep API keys in `.env` file (not tracked in git)
- Redis data persists in Docker volumes
- Add authentication before deploying to production
- Consider rate limiting for public APIs

## Development

### Running with uv

1. **Install dependencies**
   ```bash
   uv sync
   ```

2. **Start Redis**
   ```bash
   redis-server
   ```

3. **Run the server**
   ```bash
   uv run fastapi run api.py
   ```

The server will reload automatically on code changes.

## Performance

- **Video Generation**: Each 5-second clip takes ~30-60 seconds to generate
- **Audio Synthesis**: ~2-3 seconds for 15-second voiceover
- **Script Generation**: ~3-5 seconds with Qwen LLM
- **Total Pipeline**: ~2-4 minutes for a complete 15-second video (3 clips)

## Troubleshooting

### Common Issues

**Redis Connection Failed**
```bash
# Check if Redis is running
docker ps | grep redis
# View Redis logs
docker logs framestorm-redis
```

**API Key Errors**
- Verify `.env` file exists and contains valid keys
- Check key quotation formatting (no quotes needed)
- Ensure environment variables are loaded in Docker Compose

**Video Generation Timeout**
- Alibaba Dashscope may experience high load
- Implement retry logic or increase timeout values
- Consider caching generated clips for reuse

## Potential Improvements

- Support for multiple LLM providers (OpenAI, Anthropic)
- Video effects and transitions
- Batch generation endpoint
- Web UI
- Automatic subtitles
- Multi-language support
- Background music
- Analytics

## License

MIT License
