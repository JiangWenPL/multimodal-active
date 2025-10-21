import requests
import os
import base64
import logging
import random
import re
import json

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def create_query(image_path: str, num_candidates: int, frame_idx: int, total_frames: int, prev_frontier: bool = False):
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
                "```json\n"
                + json_str
                + "\n```"
                + "Do not cut off the JSON and generate the full JSON. "
            )
        },
        {
            "role": "user",
            "content": "I have a bird-eye view image of a scene. The goal is to plan a long-term exploration mission for a robot to traverse the area. Please analyze the attached image and provide the exploration plan first and then an exploration target in the specified JSON format.",
            "images": [base64_image],
        },
    ]

    query_dict = {
        "model": "llava",
        "messages": messages,
    }
    return query_dict

class ResponseError(Exception):
  """
  Common class for response errors.
  """

  def __init__(self, error: str, status_code: int = -1):
    try:
      # try to parse content as JSON and extract 'error'
      # fallback to raw content if JSON parsing fails
      error = json.loads(error).get('error', error)
    except json.JSONDecodeError:
      ...

    super().__init__(error)
    self.error = error
    'Reason for the error.'

    self.status_code = status_code
    'HTTP status code of the response.'

  def __str__(self) -> str:
    return f'{self.error} (status code: {self.status_code})'

def find_avaliable_api_url():
    for api_url in ["http://172.17.0.1:11334", "http://158.130.109.210:11234"]:
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                logging.info(f"Using the API URL: {api_url}")
                return api_url
        except requests.RequestException:
            logging.warning(f"API URL {api_url} is not available.")
    raise ConnectionError("No API URL is available.")


class LLavaFrontierSelection:
    
    def __init__(self):
        self.api_url = find_avaliable_api_url()
    
    def __call__(self, img_path: str, num_candidates: int, frame_idx: int, total_frames: int, prev_frontier: bool = False,
                 request_save_path: str = None):
        query_dict = create_query(img_path, num_candidates, frame_idx, total_frames, prev_frontier=prev_frontier)
        logging.info(f"System prompt: {query_dict['messages'][0]}")
        if request_save_path:
            with open(request_save_path, "w") as f:
                json.dump(query_dict["messages"][0], f)
        # call the resutful API of ollma:
        response = requests.post(f"{self.api_url}/api/chat", json=query_dict)
        response.raise_for_status()

        response_contents = []
        for line in response.iter_lines():
            part = json.loads(line)
            if err := part.get('error'):
                raise ResponseError(err)
            response_contents.append(part)
        
        message_str = ''.join([part['message']['content'] for part in response_contents])
        logging.info(f"VLM responded the following: {message_str}")

        pattern = r"\s*(\{.*?\})\s*" 
        match = re.search(pattern, message_str, re.DOTALL)
        response_js_str = max(match.groups(), key=len) # match the longest string

        selection_dict = json.loads(response_js_str, strict=False)
        assert "target" in selection_dict, "Target not found in the response"
        assert type(selection_dict['target']) == int, "Target cannot be parsed"
        assert selection_dict["target"] < num_candidates, "Target exceed number of candidates"

        return selection_dict, message_str 
    


if __name__ == "__main__":
    vlm = LLavaFrontierSelection()
    # vlm = VLMFrontierSelection()
    out = vlm(img_path="experiments/GaussianSLAM/Ribera-debug-vlm-llava/vlm-croped-2.png",
              num_candidates=2,
              frame_idx=2,
              total_frames=2000,
              prev_frontier=False,
              )
    breakpoint()