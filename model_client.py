"""
Model client abstraction for ADLM pipeline.

Supports two providers:
  - api:   Google API (Gemini + Gemma models) via google-genai SDK
  - local: HuggingFace transformers on local GPU
"""

import os
import json
import re
import time


def create_client(cfg):
    if cfg.model.provider == "api":
        return APIClient(cfg)
    elif cfg.model.provider == "local":
        return LocalClient(cfg)
    elif cfg.model.provider == "bedrock-mantle":
        return BedrockMantleClient(cfg)
    elif cfg.model.provider == "bedrock":
        return BedrockConverseClient(cfg)
    else:
        raise ValueError(f"Unknown provider: {cfg.model.provider}")


class APIClient:
    def __init__(self, cfg):
        from google import genai
        self.genai = genai
        self.types = genai.types
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = cfg.model.name
        self.temperature = cfg.model.temperature

    def generate(self, system_prompt, parts, schema):
        """Generate content from multimodal parts.

        Args:
            system_prompt: System instruction text
            parts: List of dicts, each either {"type": "text", "text": "..."}
                   or {"type": "image", "path": "/path/to/img.jpg"}
            schema: JSON schema dict for structured output

        Returns:
            dict with keys: data (parsed JSON), prompt_tokens, output_tokens, thinking_tokens
        """
        api_parts = []
        for p in parts:
            if p["type"] == "text":
                api_parts.append(self.types.Part(text=p["text"]))
            elif p["type"] == "image":
                with open(p["path"], "rb") as f:
                    api_parts.append(self.types.Part(
                        inline_data=self.types.Blob(data=f.read(), mime_type="image/jpeg")
                    ))

        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=self.types.Content(parts=api_parts),
                    config=self.types.GenerateContentConfig(
                        temperature=self.temperature,
                        response_mime_type="application/json",
                        response_schema=schema,
                        system_instruction=system_prompt
                    )
                )
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = self._parse_retry_delay(err_str, default=15 * (attempt + 1))
                    print(f"  Rate limited, waiting {wait:.0f}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting")

        data = self._parse_json(resp.text)

        return {
            "data": data,
            "prompt_tokens": resp.usage_metadata.prompt_token_count,
            "output_tokens": resp.usage_metadata.candidates_token_count,
            "thinking_tokens": getattr(resp.usage_metadata, 'thoughts_token_count', 0)
        }

    @staticmethod
    def _parse_retry_delay(err_str, default=15):
        match = re.search(r'retry(?:Delay|_delay|In)["\s:]+(\d+)', err_str, re.IGNORECASE)
        if match:
            return int(match.group(1)) + 2
        match = re.search(r'Please retry in ([\d.]+)s', err_str)
        if match:
            return float(match.group(1)) + 2
        return default

    @staticmethod
    def _parse_json(text):
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            brace = text.find('{')
            if brace >= 0:
                bracket_count = 0
                for i in range(brace, len(text)):
                    if text[i] == '{':
                        bracket_count += 1
                    elif text[i] == '}':
                        bracket_count -= 1
                    if bracket_count == 0:
                        try:
                            return json.loads(text[brace:i + 1])
                        except json.JSONDecodeError:
                            break
            return {"events": [], "new_characters": [], "_raw": text}


class LocalClient:
    def __init__(self, cfg):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(cfg.model.local.dtype, torch.bfloat16)
        model_name = cfg.model.name

        print(f"Loading local model: {model_name} (dtype={cfg.model.local.dtype}, device_map={cfg.model.local.device_map})")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=cfg.model.local.device_map,
        )
        self.temperature = cfg.model.temperature
        self.max_new_tokens = cfg.model.local.max_new_tokens
        self.model_name = model_name

    def generate(self, system_prompt, parts, schema):
        from PIL import Image

        content_parts = []
        images = []
        for p in parts:
            if p["type"] == "text":
                content_parts.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                img = Image.open(p["path"]).convert("RGB")
                images.append(img)
                content_parts.append({"type": "image", "image": img})

        json_instruction = (
            "\n\nYou MUST respond with valid JSON only, no markdown fencing. "
            "Follow this schema:\n" + json.dumps(schema, indent=2)
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt + json_instruction}]},
            {"role": "user", "content": content_parts},
        ]

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        prompt_tokens = inputs["input_ids"].shape[1]

        gen_kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": True}
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_ids = output_ids[0][prompt_tokens:]
        output_tokens = len(new_ids)
        raw_text = self.processor.decode(new_ids, skip_special_tokens=True)

        data = self._parse_json(raw_text)

        return {
            "data": data,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": 0
        }

    def _parse_json(self, text):
        fenced = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            brace = text.find('{')
            if brace >= 0:
                bracket_count = 0
                for i in range(brace, len(text)):
                    if text[i] == '{':
                        bracket_count += 1
                    elif text[i] == '}':
                        bracket_count -= 1
                    if bracket_count == 0:
                        try:
                            return json.loads(text[brace:i + 1])
                        except json.JSONDecodeError:
                            break
            return {"events": [], "new_characters": [], "_raw": text}


class BedrockMantleClient:
    """AWS Bedrock via bedrock-mantle endpoint (OpenAI-compatible API).

    Used for models like google.gemma-4-31b that are only on bedrock-mantle.
    """

    def __init__(self, cfg):
        import boto3
        from aws_bedrock_token_generator import BedrockTokenGenerator
        from openai import OpenAI

        region = cfg.model.get("region", "us-east-1")
        session = boto3.Session()
        credentials = session.get_credentials().get_frozen_credentials()
        generator = BedrockTokenGenerator()
        token = generator.get_token(credentials=credentials, region=region)

        self.client = OpenAI(
            base_url=f"https://bedrock-mantle.{region}.api.aws/openai/v1",
            api_key=token
        )
        self.model = cfg.model.name
        self.temperature = cfg.model.temperature

    def generate(self, system_prompt, parts, schema):
        import base64

        json_instruction = (
            "\n\nYou MUST respond with valid JSON only, no markdown fencing. "
            "Follow this schema:\n" + json.dumps(schema, indent=2)
        )

        messages = [
            {"role": "system", "content": system_prompt + json_instruction},
        ]

        user_content = []
        for p in parts:
            if p["type"] == "text":
                user_content.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                with open(p["path"], "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
        messages.append({"role": "user", "content": user_content})

        max_retries = 5
        for attempt in range(max_retries):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=4096,
                )
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "throttl" in err_str.lower():
                    wait = 15 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting")

        raw_text = resp.choices[0].message.content
        data = self._parse_json(raw_text)

        return {
            "data": data,
            "prompt_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "thinking_tokens": 0,
        }

    @staticmethod
    def _parse_json(text):
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            brace = text.find('{')
            if brace >= 0:
                bracket_count = 0
                for i in range(brace, len(text)):
                    if text[i] == '{':
                        bracket_count += 1
                    elif text[i] == '}':
                        bracket_count -= 1
                    if bracket_count == 0:
                        try:
                            return json.loads(text[brace:i + 1])
                        except json.JSONDecodeError:
                            break
            return {"events": [], "new_characters": [], "_raw": text}


class BedrockConverseClient:
    """AWS Bedrock via standard bedrock-runtime Converse API.

    Used for models like qwen.qwen3-vl-235b-a22b.
    """

    def __init__(self, cfg):
        import boto3
        region = cfg.model.get("region", "us-east-1")
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model = cfg.model.name
        self.temperature = cfg.model.temperature

    def generate(self, system_prompt, parts, schema):
        json_instruction = (
            "\n\nYou MUST respond with valid JSON only, no markdown fencing. "
            "Follow this schema:\n" + json.dumps(schema, indent=2)
        )

        content_blocks = []
        for p in parts:
            if p["type"] == "text":
                content_blocks.append({"text": p["text"]})
            elif p["type"] == "image":
                with open(p["path"], "rb") as f:
                    img_bytes = f.read()
                content_blocks.append({
                    "image": {
                        "format": "jpeg",
                        "source": {"bytes": img_bytes}
                    }
                })
        content_blocks.append({"text": json_instruction})

        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = self.client.converse(
                    modelId=self.model,
                    system=[{"text": system_prompt}],
                    messages=[{"role": "user", "content": content_blocks}],
                    inferenceConfig={
                        "maxTokens": 4096,
                        "temperature": self.temperature,
                    }
                )
                break
            except Exception as e:
                err_str = str(e)
                if "Throttling" in err_str or "429" in err_str:
                    wait = 15 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting")

        raw_text = resp["output"]["message"]["content"][0]["text"]
        data = self._parse_json(raw_text)

        usage = resp.get("usage", {})
        return {
            "data": data,
            "prompt_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "thinking_tokens": 0,
        }

    @staticmethod
    def _parse_json(text):
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            brace = text.find('{')
            if brace >= 0:
                bracket_count = 0
                for i in range(brace, len(text)):
                    if text[i] == '{':
                        bracket_count += 1
                    elif text[i] == '}':
                        bracket_count -= 1
                    if bracket_count == 0:
                        try:
                            return json.loads(text[brace:i + 1])
                        except json.JSONDecodeError:
                            break
            return {"events": [], "new_characters": [], "_raw": text}
