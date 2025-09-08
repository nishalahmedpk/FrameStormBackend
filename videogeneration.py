import requests
from http import HTTPStatus
from dashscope import VideoSynthesis
import dashscope
import os

dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

def generate_video(prompt: str,size: str , output_path: str):
    print('please wait...')
    rsp = VideoSynthesis.call(
        model='wan2.1-t2v-turbo',
        prompt=prompt,
        size=size
    )
    print(rsp)

    if rsp.status_code == HTTPStatus.OK:
        video_url = rsp.output.video_url
        print("Video generated:", video_url)

        # Download video
        response = requests.get(video_url, stream=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Video saved as: {output_path}")
        return output_path

    else:
        print('Failed, status_code: %s, code: %s, message: %s' %
              (rsp.status_code, rsp.code, rsp.message))