
import os
import base64
import logging
import random
import re
import json
from openai import OpenAI

class RepeatedQueryError(Exception):
    pass

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def generate_exploration_plan(client: OpenAI, image_path: str, num_candidates: int, frame_idx: int, total_frames: int, prev_frontier: bool = False):
    # Open image as binary
    # with open(image_path, "rb") as image_file:
    #     image_data = image_file.read()
    base64_image = encode_image(image_path)
    json_str = """\n\n{\n\"target\": 2, \"reason\": \"The target is located at an unvisited region of the image and seems to be an unvisited bedroom\"\n}\n"""

    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that can analyze images and plan a long-term goal for the exploration task of a ground robot."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "I have a bird-eye view image of a scene. The goal is to plan a long-term exploration mission for a robot to traverse the area. The robot's task is to explore the terrain efficiently, identifying important areas, potential obstacles, and unvisited areas."
                },
                {
                    "type": "text",
                    "text": "Please analyze the image and select a long-term goal from the candidates for the robot to explore the area. "
                            "Empty space doesn't always mean they are unvisited regions, sometimes it's just outside the floor plan of this scene. "
                            f"We are allowed to explore a total of {total_frames} steps and this is step {frame_idx}. "
                            "Therefore, it's better to select a space that is close to the visited regions but still unvisited and not behind the walls. "
                            "The current location of the robot is marked with the blue star(*) marker. "
                            "The visited path is painted as green lines in the image. "
                            + ("The last frontier you selected is marked with a yellow diamond(◆) shape. It is better to continue exploration around the last selection if the neighboring space is still unvisited. " if prev_frontier else "")
                            +"Note that you don't have to select the closest point to the robot, but the point that is most likely to be unvisited and important to explore. "
                            f"As you can see, there are {num_candidates} candidate points to select from. "
                            f"They are numbered from 0 to {num_candidates-1} in red color. "
                            "If you find all the goals are not necessary to explore and we should instead focus on improving existing reconstruction, please give -1 in the `target` entry of the JSON. "
                            "Please provide a detailed exploration plan and select an exploration target with reasons in the JSON format as shown below. "
                            + json_str
                },
                {
                    "type": "text",
                    "text": "Now, analyze the attached image and provide the exploration plan first and then an exploration target in the specified JSON format."
                            "Do not cut off the JSON and generate the full JSON. "
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    },
                },
            ]
        },
    ]
    # print(messages[1]['content'][1])

    # Send the request to GPT-4
    response = client.chat.completions.create(
        model="gpt-4o",  # Ensure the correct model is used (could be gpt-4-vision or similar)
        messages=messages,
        # response_format=PlanResult,
        max_completion_tokens=1000,
    )
    return response

# response = generate_exploration_plan("./experiments/debug_render-2.png")
# response = generate_exploration_plan("./experiments/ploted-image.png")
def generate_exploration_plan_system(client: OpenAI, image_path: str, num_candidates: int, frame_idx: int, total_frames: int, prev_frontier: bool = False, 
                                     request_save_path: str = None):
    # Open image as binary
    # with open(image_path, "rb") as image_file:
    #     image_data = image_file.read()
    base64_image = encode_image(image_path)
    json_str = """\n\n{\n\"target\": 2, \"reason\": \"The target is located at an unvisited region of the image and seems to be an unvisited bedroom\"\n}\n"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that can analyze images and plan a long-term goal for the exploration task of a ground robot. "
                "You will be given a bird-eye view image of a scene. The goal is to plan a long-term exploration mission for a robot to traverse the area. "
                "The robot's task is to explore the terrain efficiently, identifying important areas, potential obstacles, and unvisited areas. "
                "Please analyze the image and select a long-term goal from the candidates for the robot to explore the area. "
                "Empty space doesn't always mean they are unvisited regions, sometimes it's just outside the floor plan of this scene. "
                f"We are allowed to explore a total of {total_frames} steps and this is step {frame_idx}. "
                "Therefore, it's better to select a space that is close to the visited regions but still unvisited and not behind the walls. "
                "The current location of the robot is marked with the blue star(*) marker. "
                "The visited path is painted as green lines in the image. "
                + ("The last frontier you selected is marked with a yellow diamond(◆) shape. It is better to continue exploration around the last selection if the neighboring space is still unvisited. " if prev_frontier else "")
                + "Note that you don't have to select the closest point to the robot, but the point that is most likely to be unvisited and important to explore. "
                f"As you can see, there are {num_candidates} candidate points to select from. "
                f"They are numbered from 0 to {num_candidates-1} in red color. "
                "If you find all the goals are not necessary to explore and we should instead focus on improving existing reconstruction, please give -1 in the `target` entry of the JSON. "
                "Please provide a detailed exploration plan and select an exploration target with reasons in the JSON format as shown below. "
                + json_str
                + "Do not cut off the JSON and generate the full JSON. "
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "I have a bird-eye view image of a scene. The goal is to plan a long-term exploration mission for a robot to traverse the area. Please analyze the attached image and provide the exploration plan first and then an exploration target in the specified JSON format."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    },
                },
            ]
        },
    ]
    logging.info(f"System prompt: {messages[0]}")
    if request_save_path:
        with open(request_save_path, "w") as f:
            json.dump(messages, f)

    # Send the request to GPT-4
    response = client.chat.completions.create(
        model="gpt-4o",  # Ensure the correct model is used (could be gpt-4-vision or similar)
        messages=messages,
        # response_format=PlanResult,
        max_completion_tokens=1000,
    )
    return response

class VLMFrontierSelection:
    
    def __init__(self):
        self.client = OpenAI()
        self.last_frame_idx = -1
        self.repeat_request_cnt = 0
    
    def __call__(self, img_path: str, num_candidates: int, frame_idx: int, total_frames: int, prev_frontier: bool = False, 
                 request_save_path: str = None):
        if frame_idx - self.last_frame_idx < 2: # to avoid running out of our creidts when the robot got stuck
            self.repeat_request_cnt += 1
            if self.repeat_request_cnt > 3:
                self.repeat_request_cnt = 0
                raise RepeatedQueryError("Repeated requests to VLM. Skipping this frame.")
        self.last_frame_idx = frame_idx

        response = generate_exploration_plan_system(self.client, img_path, num_candidates, frame_idx, total_frames, prev_frontier=prev_frontier, request_save_path=request_save_path)
        message = response.choices[0].message
        logging.info(f"VLM responded the following: {message.content}")
        # pattern = r"```json\s*(\{.*?\})\s*```"
        pattern = r"\s*(\{.*?\})\s*"
        match = re.search(pattern, message.content, re.DOTALL)
        selection_dict = json.loads(match.group(1), strict=False)
        assert "target" in selection_dict, "Target not found in the response"
        assert type(selection_dict['target']) == int, "Target cannot be parsed"
        assert selection_dict["target"] < num_candidates, "Target exceed number of candidates"

        return selection_dict, message.content
