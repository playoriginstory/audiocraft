from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch

class EndpointHandler:
    def __init__(self, path=""):
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = MusicgenForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16).to("cuda")

    def __call__(self, data: dict) -> dict:
        """
        Args:
            data (dict): Contains the text prompt, vibe, style, and public domain song reference.
        """
        # Extract user inputs
        text_prompt = data.get("text_prompt")
        vibe = data.get("vibe")
        style = data.get("style")
        song_reference = data.get("song_reference")

        # Combine user inputs to form the complete prompt
        combined_prompt = f"{vibe} {style} version of {song_reference}: {text_prompt}"

        # Process the prompt
        inputs = self.processor(text=[combined_prompt], padding=True, return_tensors="pt").to("cuda")

        # Generate music
        with torch.autocast("cuda"):
            audio_output = self.model.generate(**inputs)

        # Convert to suitable format
        audio_data = audio_output[0].cpu().numpy().tolist()

        # Return generated music
        return {"generated_audio": audio_data}

# Replace with the actual path or model identifier
handler = EndpointHandler(path="path-to-your-model")
