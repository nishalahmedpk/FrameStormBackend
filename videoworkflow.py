from typing import Annotated, NotRequired, List, Tuple

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    # messages: Annotated[list, add_messages]
    # prompts: List[str] = []
    # generated_video_files: List[str] = []
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
        video_files = [os.path.join("placeholder",v) for v in video_files]
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
        video_files = [os.path.join("placeholder",v) for v in video_files]

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
