from typing import Annotated, NotRequired, List, Tuple

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, AnyMessage


class State(TypedDict):
    messages: NotRequired[Annotated[list[AnyMessage], add_messages]]
    final_video: NotRequired[str]
    project_name: str = "Viral Video Maker"
    description: str = ""
    generate_video: bool = False
    prompts: NotRequired[List[str]] = []
    script: NotRequired[str] = ""
    voiceover: NotRequired[str] = ""
    cuts: NotRequired[List[int]] = []
    audio_file: NotRequired[str] = ""
    video_files: NotRequired[List[str]] = []
    audio_duration: NotRequired[float] = 0.0
    cuts: List[Tuple[float, float, float]] = []
    blog: NotRequired[str] = ""


graph_builder = StateGraph(State)



from dotenv import load_dotenv
import os
load_dotenv()

from langchain_qwq import ChatQwen  # Qwen LLM
llm = ChatQwen(model="qwen3-coder-plus")

blog_llm = ChatQwen(model="qwen-flash")

from langchain.prompts import ChatPromptTemplate

blog_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are an expert blogger and content writer.  
Given a short user prompt, expand it into a **comprehensive, engaging blog post**.  

The blog should:  
- Be at least 800–1200 words (unless otherwise specified).  
- Start with a strong introduction that hooks the reader.  
- Be structured with clear headings and subheadings.  
- Use smooth transitions and a natural, conversational tone.  
- End with a powerful conclusion or call-to-action.  

Return only the blog text, no JSON, no explanations."""
    ),
    ("human", "{input}"),
])


def BlogGenerator(prompt: str):
    result = (blog_prompt | blog_llm).invoke({"input": prompt})
    return result


import requests
import json
import time

from videogeneration import generate_video
def VideoGenerator(state: State):
    if state["generate_video"]==False:
        print("Skipping video generation as per user request.")
        video_files = ["output0.mp4", "output1.mp4", "output2.mp4", "output3.mp4", "output4.mp4"]
        video_files = [os.path.join("dummyvids",v) for v in video_files]
        return Command(update={"video_files": video_files})
    project_name = state["project_name"]
    video_files = []
    prompts = state["prompts"]
    for i, prompt in enumerate(prompts):
        print(f"Generating video for prompt {i+1}: {prompt}")
        path = os.path.join(project_name, f"output{i}.mp4")
        generate_video(prompt, "1280*720", path)
        video_files.append(path)
    return Command(update={"video_files": video_files})


from langgraph.types import Command
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate


class ScriptOutput(BaseModel):
    script: str
    voiceover: str
prompt2 = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a skilled voiceover artist and scriptwriter specializing in creating captivating viral short videos.  
Given a user’s description, generate a **15-second voiceover script** that is expressive, engaging, and drives emotional connection.  

The voiceover should:  
- Be written exactly as it would be spoken — no stage directions, tone markers, or parenthetical notes.  
- Use natural, conversational language that flows smoothly.  
- Be the centerpiece, with the accompanying video script serving as visual support to enhance the narration.  

Return the output as JSON with keys:  
- 'voiceover': the spoken script as plain text only (no parentheses or tone labels).  
- 'script': a short description of the video visuals that would match the voiceover."""
),
    ("human", "{input}"),
])

def ScriptGenerator(state: State):
    llm_structured = llm.with_structured_output(ScriptOutput)
    result = (prompt2 | llm_structured).invoke({"input": state["description"]})
    return Command(update={"script": result.script, "voiceover": result.voiceover})



from elevenlabs.client import ElevenLabs
from moviepy import (
    VideoClip,
    VideoFileClip,
    ImageSequenceClip,
    ImageClip,
    TextClip,
    ColorClip,
    AudioFileClip,
    AudioClip,
)
import numpy as np

def audio_generator(state: State):
    if "voiceover" not in state or not state["voiceover"]:
        print("No voiceover provided, skipping audio generation.")
        return state
    
    print(f"Generating audio for voiceover: {state['voiceover']}")

    project_name = state["project_name"]
    audio_filename = os.path.join(project_name, "output_audio.mp3")

    # Initialize ElevenLabs client
    client = ElevenLabs()

    # Convert text to speech (returns a generator)
    audio_stream = client.text_to_speech.convert(
        text=state["voiceover"],
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )

    # Combine generator chunks into bytes
    audio_bytes = b"".join(audio_stream)

    # Save to file
    with open(audio_filename, "wb") as f:
        f.write(audio_bytes)

    print(f"Generated audio saved as '{audio_filename}'")

    # Obtain the length of the audio file in seconds
    from moviepy import AudioFileClip
    audio_clip = AudioFileClip(audio_filename)
    audio_duration = audio_clip.duration
    audio_clip.close()
    print(f"Audio duration: {audio_duration} seconds")

    return Command(update={"audio_file": audio_filename, "audio_duration": audio_duration})


from typing import Tuple

class PromptOutput(BaseModel):
    prompts: List[str]
    cuts: List[Tuple[float, float, float]]

prompt3 = ChatPromptTemplate.from_messages([
    ("system",
     """You are an expert cinematic storyteller and prompt engineer specializing in creating detailed, scene-by-scene prompts for video generation.  

Rules:
- The total audio length is {audio_duration} seconds.  
- Each scene must be exactly 5 seconds long, except the final one which may be shorter if the audio does not divide evenly.  
- Calculate number of scenes = ceil(audio_duration / 5).  
- For each scene, generate:
   1. A vivid cinematic description (added to 'prompts').
   2. A corresponding cut tuple (start, stop, duration in seconds) added to 'cuts'.
- The format for cuts is: (start, stop, total_duration).  
- The sum of all total_durations in 'cuts' MUST equal {audio_duration}.  
- Return ONLY valid JSON with two keys: 'prompts' and 'cuts'.""" 
    ),
    ("human", "{input}"),
])

def VideoMetadataGenerator(state: State):
    # llm1 = prompt1 | llm
    # Create a list of prompts from planner.
    if "audio_duration" not in state or not state["audio_duration"]:
        print("No audio duration available, using dummy prompts.")
        dummy_prompts = [
        "A cat playing piano in a jazz bar with lively music.",
        "A bustling street market with colorful stalls and ambient chatter.",
        ]
        return Command(update={"prompts": dummy_prompts})
    else:
        import math
        llm_structured = llm.with_structured_output(PromptOutput)
        result = (prompt3 | llm_structured).invoke({"input": state["voiceover"], "audio_duration": state["audio_duration"]})
        print(f"Generated prompts: {result.prompts}")
        print(f"Generated cuts: {result.cuts}")
        return Command(update={"prompts": result.prompts, "cuts": result.cuts})

import numpy as np
from moviepy import AudioFileClip, concatenate_audioclips, concatenate_videoclips

def combine_videos_with_audio(state: State):

    video_files = state["video_files"]
    print(video_files)
    if state["generate_video"]==False: ## THIS IS SO I DONT RUN OUT OF CREDITS
        print("Skipping video generation as per user request.")
        video_files = ["output0.mp4", "output1.mp4", "output2.mp4", "output3.mp4", "output4.mp4"]
        video_files = [os.path.join("dummyvids",v) for v in video_files]

    else:
        video_files = state["video_files"]

    if not video_files:
        print("No video files found in state — cannot combine.")
        return state
    
    audio_file = state["audio_file"]
    # audio_file = os.path.join(state["project_name"], audio_file)
    if not os.path.exists(audio_file):
        print("No audio file found — cannot combine.")
        return state

    cuts = state["cuts"]
    clips = []
    try:
        for i,cut in enumerate(cuts):
            if not os.path.exists(video_files[i]):
                print(f"File not found: {video_files[i]}")
                continue
            clip = VideoFileClip(video_files[i])
            if clip.duration is None:
                print(f"Skipping {video_files[i]}: duration is None")
                clip.close()
                continue
            trimmed_clip = clip.subclipped(0, cut[2])
            clips.append(trimmed_clip)

        if not clips:
            print("No valid video clips loaded.")
            return state

        final_clip = concatenate_videoclips(clips, method="compose")

        audio_clip = AudioFileClip(audio_file)

        # If audio is shorter, trim the video to audio duration (remove silence padding)
        if audio_clip.duration < final_clip.duration:
            print(f"Trimming video from {final_clip.duration} to audio duration {audio_clip.duration}")
            final_clip = final_clip.subclipped(0, audio_clip.duration)

        # If audio longer, trim audio to video duration
        # elif audio_clip.duration > final_clip.duration:
        #     audio_clip = audio_clip.subclip(0, final_clip.duration)

        final_clip = final_clip.with_audio(audio_clip)

        output_file = os.path.join(state["project_name"], "final_video.mp4")
        final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

        print(f"Final video saved as {output_file}")
        return Command(update={"final_video": output_file})

    except Exception as e:
        print(f"Error combining videos: {e}")
        return state

    finally:
        for c in clips:
            c.close()


graph_builder = StateGraph(State)
graph_builder.add_node("ScriptGenerator", ScriptGenerator)
graph_builder.add_node("audio_generator", audio_generator)
graph_builder.add_node("VideoMetadataGenerator", VideoMetadataGenerator)
graph_builder.add_node("VideoGenerator", VideoGenerator)
graph_builder.add_node("combine_videos_with_audio", combine_videos_with_audio)
graph_builder.add_edge(START, "ScriptGenerator")
graph_builder.add_edge("ScriptGenerator", "audio_generator")
graph_builder.add_edge("audio_generator", "VideoMetadataGenerator")  # Changed to match function name in the
graph_builder.add_edge("VideoMetadataGenerator", "VideoGenerator")
graph_builder.add_edge("VideoGenerator", "combine_videos_with_audio")
graph_builder.add_edge("combine_videos_with_audio", END)
graph = graph_builder.compile()

# graph.invoke(State(description=""" Create a viral video for promoting BTIS Pilani""")) #Wrap this in a function to invoke the graph

chatgraph = StateGraph(State)

audio_generation_template = """
You are an audio generation specialist for video projects.
Your primary responsibility is to determine if new audio needs to be generated and manage voiceover creation.

Rules for audio generation:

1. **Audio Analysis**:
    - Check if the user is requesting a new voiceover or audio modification
    - Verify if the current `audio_file` exists and matches the project requirements
    - Analyze if the `voiceover` text has been updated or changed

2. **Audio Generation Decision**:
    - Generate new audio ONLY if:
        * User explicitly requests new voiceover
        * The voiceover text has been modified
        * No audio file exists but voiceover text is provided
        * User wants to change the voice or audio style
    - Do NOT generate audio if:
        * User is only asking about video editing or rearrangement
        * Audio already exists and matches current voiceover
        * No voiceover text changes are requested

3. **Tool Usage**:
    - Use `generate_new_audio` tool when audio generation is needed
    - Always include the current voiceover text and project_name
    - Update the audio_file and audio_duration in the state

4. **Communication**:
    - Clearly explain your decision whether to generate audio or not
    - If generating audio, inform the user about the process
    - If skipping, explain why audio generation is not needed

Current state:
    "project_name": {project_name}
    "voiceover": {voiceover}
    "audio_duration": {audio_duration}
    "audio_file": {audio_file}

Only use the `generate_new_audio` tool if audio generation is actually required.
"""

video_generation_template = """
You are a video generation specialist for video projects.
Your responsibility is to determine if new video clips need to be generated based on project requirements.

Rules for video generation:

1. **Video Analysis**:
    - Check the current `video_files` list and their availability
    - Compare `audio_duration` with total video duration (each video is 5 seconds)
    - Assess if user is requesting new video content or additional clips

2. **Video Generation Decision**:
    - Generate new video ONLY if:
        * User explicitly requests new video content
        * Audio duration exceeds available video duration (need more clips)
        * Specific new scenes or prompts are requested
        * Current videos don't match the project theme
    - Do NOT generate video if:
        * Sufficient videos already exist for the audio duration
        * User is only asking about audio or rearrangement
        * No new content is specifically requested

3. **Tool Usage**:
    - Use `generate_new_video` tool when new video generation is needed
    - Provide descriptive prompts for video content
    - Use appropriate video size (default: "1280*720")
    - Update video_files list with new video paths

4. **Communication**:
    - Explain your decision whether to generate videos or not
    - If generating, describe what type of video will be created
    - If skipping, explain why video generation is not needed

Current state:
    "video_files": {video_files}
    "audio_duration": {audio_duration}
    "project_name": {project_name}

Only use the `generate_new_video` tool if video generation is actually required.
"""

clip_rearrangement_template = """
You are a video clip modification specialist for video projects.
Your responsibility is to organize, trim, arrange, add, remove, and modify video clips according to user requirements.

Rules for clip modification:

1. **Modification Analysis**:
    - Check if user wants to modify video timing, order, duration, add new clips, or remove existing ones
    - Analyze current `cuts` and `video_files` to understand existing arrangement
    - Verify what specific changes are needed (reorder, trim, add, remove, or adjust timing)

2. **Modification Decision**:
    - Modify clips ONLY if:
        * User explicitly requests video reordering
        * User wants to add new video clips to the sequence
        * User wants to remove or delete specific clips
        * Timing adjustments are needed for audio sync
        * Clip durations need to be modified or trimmed
        * User wants to change clip order or sequence
        * User wants to remove or shorten specific scenes
    - Do NOT modify if:
        * User is only asking about audio or new video generation
        * Current arrangement is satisfactory
        * No specific changes are requested

3. **Tool Usage**:
    - Use `rearrange_clips` tool when modification is needed
    - Ensure cut durations are valid (0 to 5 seconds per clip)
    - Provide proper video file paths and cut specifications
    - Handle clip additions by updating video_files list
    - Handle clip removals by excluding from rearrangement
    - Save the final modified video

4. **Communication**:
    - Explain your decision whether to modify clips or not
    - If modifying, describe the changes being made (add/remove/reorder/trim)
    - If skipping, explain why modification is not needed

Current state:
    "video_files": {video_files}
    "cuts": {cuts}
    "project_name": {project_name}

Only use the `rearrange_clips` tool if clip modification is actually required.
"""



audio_generation_prompt = ChatPromptTemplate.from_messages(
    [("system", audio_generation_template), ("human", "{input}")]
)

video_generation_prompt = ChatPromptTemplate.from_messages(
    [("system", video_generation_template), ("human", "{input}")]
)

clip_rearrangement_prompt = ChatPromptTemplate.from_messages(
    [("system", clip_rearrangement_template), ("human", "{input}")]
)


from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

@tool
def generate_new_audio(new_voiceover: str, project_name: str, tool_call_id: Annotated[str, InjectedToolCallId] = None) -> str:
    """Generate new audio using ElevenLabs TTS and save to the audio file."""
    audio_filename = os.path.join(project_name, "output_audio.mp3")

    # Initialize ElevenLabs client
    client = ElevenLabs()

    # Convert text to speech (returns a generator)
    audio_stream = client.text_to_speech.convert(
        text=new_voiceover,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )

    # Combine generator chunks into bytes
    audio_bytes = b"".join(audio_stream)

    # Save to file
    with open(audio_filename, "wb") as f:
        f.write(audio_bytes)

    print(f"Generated audio saved as '{audio_filename}'")

    # Obtain the length of the audio file in seconds
    from moviepy import AudioFileClip
    audio_clip = AudioFileClip(audio_filename)
    audio_duration = audio_clip.duration
    audio_clip.close()
    print(f"Audio duration: {audio_duration} seconds")

    return Command(update={"audio_file": audio_filename, "audio_duration": audio_duration, "messages": [
        ToolMessage(
            content="Audio updated successfully ✅",
            tool_call_id=tool_call_id,
        )
    ]})

@tool
def rearrange_clips(videos: List[str], new_cuts: List[Tuple[float, float, float]], project_name: str, audio_file: str, tool_call_id: Annotated[str, InjectedToolCallId] = None) -> str:
    """Rearrange video clips based on new cuts."""
    print("Entered Rearrange Clips Tool")
    try:
        clips = []
        for i, cut in enumerate(new_cuts):
            if not os.path.exists(videos[i]):
                print(f"File not found: {videos[i]}")
                continue
            clip = VideoFileClip(videos[i])
            if clip.duration is None:
                print(f"Skipping {videos[i]}: duration is None")
                clip.close()
                continue
            trimmed_clip = clip.subclipped(0, cut[2])
            clips.append(trimmed_clip)

        if not clips:
            print("No valid video clips loaded.")
            return "No valid video clips loaded."

        final_clip = concatenate_videoclips(clips, method="compose")

        # output_file = os.path.join(project_name, "final_video.mp4")
        # final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

        audio_clip = AudioFileClip(audio_file)

        # If audio is shorter, trim the video to audio duration (remove silence padding)
        if audio_clip.duration < final_clip.duration:
            print(f"Trimming video from {final_clip.duration} to audio duration {audio_clip.duration}")
            final_clip = final_clip.subclipped(0, audio_clip.duration)

        # If audio longer, trim audio to video duration
        # elif audio_clip.duration > final_clip.duration:
        #     audio_clip = audio_clip.subclip(0, final_clip.duration)

        final_clip = final_clip.with_audio(audio_clip)

        output_file = os.path.join(project_name, "final_video.mp4")
        final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")


        print(f"Final video saved as {output_file}")
        return Command(update={"final_video": output_file, "messages": [
            ToolMessage(
                content="Video rearranged successfully ✅",
                tool_call_id=tool_call_id,
            )
        ]})

    except Exception as e:
        print(f"Error rearranging videos: {e}")
        return f"Error rearranging videos: {e}"

    finally:
        for c in clips:
            c.close()

@tool
def generate_new_video(video_files: List[str],filename: str,prompt: str, size: str, project_name: str, tool_call_id: Annotated[str, InjectedToolCallId] = None) -> str:
    """Generate a new video clip based on the prompt."""
    print("Entered Generate New Video Tool")
    from videogeneration import generate_video
    #ERROR
    # path = os.path.join(project_name, f"{filename}.mp4")
    path = filename
    generate_video(prompt, size, path)
    video_files.append(path)
    return Command(update={"video_files": video_files, "messages": [
        ToolMessage(
            content="New video generated successfully ✅",
            tool_call_id=tool_call_id,
        )
    ]})

checkpoint_saver = InMemorySaver()

def AudioGenerationAgent(state: State):
    llm_chain = audio_generation_prompt | llm.bind_tools([generate_new_audio])
    llm_response = llm_chain.invoke({"input": state.get("messages", []),
                                    "project_name": state.get("project_name", "default_project"),
                                    "voiceover": state.get("voiceover", ""),
                                    "audio_duration": state.get("audio_duration", 0.0),
                                    "audio_file": state.get("audio_file", "")})
    return {"messages": llm_response}

def VideoGenerationAgent(state: State):
    llm_chain = video_generation_prompt | llm.bind_tools([generate_new_video])
    llm_response = llm_chain.invoke({"input": state.get("messages", []),
                                    "video_files": state.get("video_files", []),
                                    "audio_duration": state.get("audio_duration", 0.0),
                                    "project_name": state.get("project_name", "default_project")})
    return {"messages": llm_response}

def ClipRearrangementAgent(state: State):
    llm_chain = clip_rearrangement_prompt | llm.bind_tools([rearrange_clips])
    llm_response = llm_chain.invoke({"input": state.get("messages", []),
                                    "video_files": state.get("video_files", []),
                                    "cuts": state.get("cuts", []),
                                    "project_name": state.get("project_name", "default_project"),
                                    "audio_file": state.get("audio_file", "")})
    return {"messages": llm_response}

def route_audio_tools(state: State):
    messages = state.get("messages", [])
    if messages:
        ai_message = messages[-1]
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "audio_tools"
    return "VideoGenerationAgent"

def route_video_tools(state: State):
    messages = state.get("messages", [])
    if messages:
        ai_message = messages[-1]
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "video_tools"
    return "ClipRearrangementAgent"

def route_clip_tools(state: State):
    messages = state.get("messages", [])
    if messages:
        ai_message = messages[-1]
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "clip_tools"
    return END

chatgraph.add_node("AudioGenerationAgent", AudioGenerationAgent)
chatgraph.add_node("VideoGenerationAgent", VideoGenerationAgent)
chatgraph.add_node("ClipRearrangementAgent", ClipRearrangementAgent)
chatgraph.add_node("audio_tools", ToolNode([generate_new_audio]))
chatgraph.add_node("video_tools", ToolNode([generate_new_video]))
chatgraph.add_node("clip_tools", ToolNode([rearrange_clips]))

chatgraph.add_conditional_edges("AudioGenerationAgent", route_audio_tools, {
    "audio_tools": "audio_tools", 
    "VideoGenerationAgent": "VideoGenerationAgent"
})
chatgraph.add_conditional_edges("VideoGenerationAgent", route_video_tools, {
    "video_tools": "video_tools",
    "ClipRearrangementAgent": "ClipRearrangementAgent"
})
chatgraph.add_conditional_edges("ClipRearrangementAgent", route_clip_tools, {
    "clip_tools": "clip_tools",
    END: END
})

chatgraph.add_edge("audio_tools", "VideoGenerationAgent")
chatgraph.add_edge("video_tools", "ClipRearrangementAgent")
chatgraph.add_edge("clip_tools", END)
chatgraph.add_edge(START, "AudioGenerationAgent")

chatgraph = chatgraph.compile(checkpointer=checkpoint_saver)