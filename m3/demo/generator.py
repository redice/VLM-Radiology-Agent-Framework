import logging
import tempfile
import os
import time
from copy import deepcopy

import torch
from experts.expert_monai_brats import ExpertBrats
from experts.expert_monai_vista3d import ExpertVista3D
from experts.expert_torchxrayvision import ExpertTXRV
from experts.utils import ImageCache, get_modality, get_slice_filenames, image_to_data_url, load_image
from huggingface_hub import snapshot_download
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
relativeDir = os.getenv("RELATIVE_DIRECTORY")

IMG_URLS_OR_PATHS = {
    "CT Sample 1": "https://developer.download.nvidia.com/assets/Clara/monai/samples/ct_liver_0.nii.gz",
    "CT Sample 2": "https://developer.download.nvidia.com/assets/Clara/monai/samples/ct_sample.nii.gz",
    "MRI Sample 1": [
        "https://developer.download.nvidia.com/assets/Clara/monai/samples/mri_Brats18_2013_31_1_t1.nii.gz",
        "https://developer.download.nvidia.com/assets/Clara/monai/samples/mri_Brats18_2013_31_1_t1ce.nii.gz",
        "https://developer.download.nvidia.com/assets/Clara/monai/samples/mri_Brats18_2013_31_1_t2.nii.gz",
        "https://developer.download.nvidia.com/assets/Clara/monai/samples/mri_Brats18_2013_31_1_flair.nii.gz",
    ],
    "Chest X-ray Sample 1": "https://developer.download.nvidia.com/assets/Clara/monai/samples/cxr_00026451_030.jpg",
    "Chest X-ray Sample 2": "https://developer.download.nvidia.com/assets/Clara/monai/samples/cxr_00029943_005.jpg",
}

MODEL_CARDS = "Here is a list of available expert models:\n<BRATS(args)> Modality: MRI, Task: segmentation, Overview: A pre-trained model for volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data, Accuracy: Tumor core (TC): 0.8559 - Whole tumor (WT): 0.9026 - Enhancing tumor (ET): 0.7905 - Average: 0.8518, Valid args are: None\n<VISTA3D(args)> Modality: CT, Task: segmentation, Overview: domain-specialized interactive foundation model developed for segmenting and annotating human anatomies with precision, Accuracy: 127 organs: 0.792 Dice on average, Valid args are: 'everything', 'hepatic tumor', 'pancreatic tumor', 'lung tumor', 'bone lesion', 'organs', 'cardiovascular', 'gastrointestinal', 'skeleton', or 'muscles'\n<VISTA2D(args)> Modality: cell imaging, Task: segmentation, Overview: model for cell segmentation, which was trained on a variety of cell imaging outputs, including brightfield, phase-contrast, fluorescence, confocal, or electron microscopy, Accuracy: Good accuracy across several cell imaging datasets, Valid args are: None\n<CXR(args)> Modality: chest x-ray (CXR), Task: classification, Overview: pre-trained model which are trained on large cohorts of data, Accuracy: Good accuracy across several diverse chest x-rays datasets, Valid args are: None\nGive the model <NAME(args)> when selecting a suitable expert model.\n"

SYS_PROMPT = None  # set when the script initializes

CACHED_IMAGES = ImageCache(cache_dir=tempfile.mkdtemp())


class ChatHistory:
    """Class to store the chat history"""

    def __init__(self):
        """
        Messages are stored as a list, with a sample format:

        messages = [
        # --------------- Below is the previous prompt from the user ---------------
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in the image? <image>"
                },
                {
                    "type": "image_path",
                    "image_path": image_path
                }
            ]
        },
        # --------------- Below is the answer from the previous completion ---------------
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": answer1,
                }
            ]
        },
        ]
        """
        self.messages = []
        self.last_prompt_with_image = None

    def append(self, prompt_or_answer, image_path=None, role="user"):
        """
        Append a new message to the chat history.

        Args:
            prompt_or_answer (str): The text prompt from human or answer from AI to append.
            image_url (str): The image file path to append.
            slice_index (int): The slice index for 3D images.
            role (str): The role of the message. Default is "user". Other option is "assistant" and "expert".
        """
        new_contents = [
            {
                "type": "text",
                "text": prompt_or_answer,
            }
        ]
        if image_path is not None:
            new_contents.append(
                {
                    "type": "image_path",
                    "image_path": image_path,
                }
            )
            self.last_prompt_with_image = prompt_or_answer

        self.messages.append({"role": role, "content": new_contents})

    def get_html(self, show_all: bool = False, sys_msgs_to_hide: list | None = None):
        """Returns the chat history as an HTML string to display"""
        history = []

        for message in self.messages:
            role = message["role"]
            contents = message["content"]
            history_text_html = ""
            for content in contents:
                if content["type"] == "text":
                    history_text_html += colorcode_message(
                        text=content["text"], show_all=show_all, role=role, sys_msgs_to_hide=sys_msgs_to_hide
                    )
                else:
                    image_paths = (
                        content["image_path"] if isinstance(content["image_path"], list) else [content["image_path"]]
                    )
                    for image_path in image_paths:
                        history_text_html += colorcode_message(
                            data_url=image_to_data_url(image_path, max_size=(300, 300)), show_all=True, role=role
                        )  # always show the image
            history.append(history_text_html)
        return "<br>".join(history)


class SessionVariables:
    """Class to store the session variables"""

    def __init__(self):
        """Initialize the session variables"""
        self.sys_prompt = SYS_PROMPT
        self.sys_msg = MODEL_CARDS
        self.use_model_cards = True
        self.slice_index = None  # Slice index for 3D images
        self.image_url = None  # Image URL to the image on the web
        self.backup = {}  # Cached varaiables from previous messages for the current conversation
        self.axis = 2
        self.top_p = 0.9
        self.temperature = 0.0
        self.max_tokens = 1024
        self.temp_working_dir = None
        self.idx_range = (None, None)
        self.interactive = False
        self.sys_msgs_to_hide = []
        self.modality_prompt = "Auto"
        self.img_urls_or_paths = IMG_URLS_OR_PATHS

    def restore_from_backup(self, attr):
        """Retrieve the attribute from the backup"""
        attr_val = self.backup.get(attr, None)
        if attr_val is not None:
            self.__setattr__(attr, attr_val)


def new_session_variables(**kwargs):
    """Create a new session variables but keep the conversation mode"""
    if len(kwargs) == 0:
        return SessionVariables()
    sv = SessionVariables()
    for key, value in kwargs.items():
        if sv.__getattribute__(key) != value:
            sv.__setattr__(key, value)
    return sv


class M3Generator:
    """Class to generate M3 responses"""

    def __init__(self, source="huggingface", model_path="MONAI/Llama3-VILA-M3-8B", conv_mode="llama_3"):
        """Initialize the M3 generator"""
        global SYS_PROMPT
        self.source = source
        if source == "local" or source == "huggingface":
            # TODO: allow setting the device
            disable_torch_init()
            self.conv_mode = conv_mode
            if source == "huggingface":
                model_path = snapshot_download(model_path)
            model_name = get_model_name_from_path(model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path, model_name
            )
            logger.info(f"Model {model_name} loaded successfully. Context length: {self.context_len}")
            SYS_PROMPT = conv_templates[self.conv_mode].system  # only set once
        else:
            raise NotImplementedError(f"Source {source} is not supported.")

    def generate_response_local(
        self,
        messages: list = [],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.9,
        system_prompt: str | None = None,
    ):
        """Generate the response"""
        logger.debug(f"Generating response with {len(messages)} messages")
        images = []

        conv = conv_templates[self.conv_mode].copy()
        if system_prompt is not None:
            conv.system = system_prompt
        user_role = conv.roles[0]
        assistant_role = conv.roles[1]

        for message in messages:
            role = user_role if message["role"] == "user" else assistant_role
            prompt = ""
            for content in message["content"]:
                if content["type"] == "text":
                    prompt += content["text"]
                if content["type"] == "image_path":
                    image_paths = (
                        content["image_path"] if isinstance(content["image_path"], list) else [content["image_path"]]
                    )
                    for image_path in image_paths:
                        images.append(load_image(image_path))
            conv.append_message(role, prompt)

        if conv.sep_style == SeparatorStyle.LLAMA_3:
            conv.append_message(assistant_role, "")  # add "" to the assistant message

        prompt_text = conv.get_prompt()
        logger.debug(f"Prompt input: {prompt_text}")

        if len(images) > 0:
            logger.debug(f"33333")
            images_tensor = process_images(images, self.image_processor, self.model.config).to(
                self.model.device, dtype=torch.float16
            )
        images_input = [images_tensor] if len(images) > 0 else None
        logger.debug(f"44444")
        input_ids = (
            tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )
        logger.debug(f"55555")
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        start_time = time.time()
        logger.debug(f"66666")
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_input,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=1,
                max_new_tokens=max_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.eos_token_id,
                min_new_tokens=2,
            )
        end_time = time.time()
        logger.debug(f"Time taken to generate {len(output_ids[0])} tokens: {end_time - start_time:.2f} seconds")
        logger.debug(f"Tokens per second: {len(output_ids[0]) / (end_time - start_time):.2f}")

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        logger.debug(f"Assistant: {outputs}")

        return outputs

    def generate_response(self, **kwargs):
        """Generate the response"""
        if self.source == "local" or self.source == "huggingface":
            return self.generate_response_local(**kwargs)
        raise NotImplementedError(f"Source {self.source} is not supported.")

    def squash_expert_messages_into_user(self, messages: list):
        """Squash consecutive expert messages into a single user message."""
        logger.debug("Squashing expert messages into user messages")
        messages = deepcopy(messages)  # Create a deep copy to avoid modifying the original list

        i = 0
        while i < len(messages):
            if messages[i]["role"] == "expert":
                messages[i]["role"] = "user"
                j = i + 1
                while j < len(messages) and messages[j]["role"] == "expert":
                    messages[i]["content"].extend(messages[j]["content"])  # Append the content directly
                    j += 1
                del messages[i + 1 : j]  # Remove all the squashed expert messages

            i += 1

        return messages

    def process_prompt(self, prompt, sv, chat_history):
        """Process the prompt and return the result. Inputs/outputs are the gradio components."""
        logger.debug(f"==> Process the image and return the result")
        logger.debug(f"first image: {sv.image_url}")
        if sv.temp_working_dir is None:
            sv.temp_working_dir = tempfile.mkdtemp()

        if sv.modality_prompt == "Auto":
            modality = get_modality(sv.image_url, text=prompt)
        else:
            modality = sv.modality_prompt
        mod_msg = f"This is a {modality} image.\n" if modality != "Unknown" else ""
        logger.debug(f"mod_msg: {mod_msg}")

        model_cards = sv.sys_msg if sv.use_model_cards else ""

        img_file = sv.image_url
        logger.debug(f"img_file: {img_file}")

        if isinstance(img_file, str):
            logger.debug(f"single image")
            if "<image>" not in prompt:
                _prompt = model_cards + "<image>" + mod_msg + prompt
                sv.sys_msgs_to_hide.append(model_cards + "<image>" + mod_msg)
            else:
                _prompt = model_cards + mod_msg + prompt
                if model_cards + mod_msg != "":
                    sv.sys_msgs_to_hide.append(model_cards + mod_msg)

            if img_file.endswith(".nii.gz"):  # Take the specific slice from a volume
                chat_history.append(
                    _prompt,
                    image_path=os.path.join(CACHED_IMAGES.dir(), get_slice_filenames(img_file, sv.slice_index)),
                )
            else:
                chat_history.append(_prompt, image_path=img_file)
        elif isinstance(img_file, list):
            logger.debug(f"multiple images")
            # multi-modal images
            prompt = (
                prompt.replace("<image>", "") if "<image>" in prompt else prompt
            )  # remove the image token if it's in the prompt
            special_token = "T1(contrast enhanced): <image>, T1: <image>, T2: <image>, FLAIR: <image> "
            mod_msg = f"These are different {modality} modalities.\n"
            _prompt = model_cards + special_token + mod_msg + prompt
            image_paths = [os.path.join(relativeDir, get_slice_filenames(f, sv.slice_index)) for f in img_file]
            chat_history.append(_prompt, image_path=image_paths)
            sv.sys_msgs_to_hide.append(model_cards + special_token + mod_msg)
        elif img_file is None:
            # text-only prompt
            chat_history.append(prompt)  # no image token
        else:
            raise ValueError(f"Invalid image file: {img_file}")

        outputs = self.generate_response(
            messages=self.squash_expert_messages_into_user(chat_history.messages),
            max_tokens=sv.max_tokens,
            temperature=sv.temperature,
            top_p=sv.top_p,
            system_prompt=sv.sys_prompt,
        )

        chat_history.append(outputs, role="assistant")

        # check the message mentions any expert model
        expert = None

        for expert_model in [ExpertTXRV, ExpertVista3D, ExpertBrats]:
            expert = expert_model() if expert_model().mentioned_by(outputs) else None
            if expert:
                break

        if expert:
            logger.debug(f"Expert model {expert.__class__.__name__} is being called to process {sv.image_url}.")
            try:
                if sv.image_url is None:
                    logger.debug(
                        "Image URL is None. Try restoring the image URL from the backup to continue expert processing."
                    )
                    sv.restore_from_backup("image_url")
                    sv.restore_from_backup("slice_index")
                text_output, seg_image, instruction = expert.run(
                    image_url=sv.image_url,
                    input=outputs,
                    output_dir=sv.temp_working_dir,
                    img_file=CACHED_IMAGES.get(sv.image_url, None, list_return=True),
                    slice_index=sv.slice_index,
                    prompt=prompt,
                )
            except Exception as e:
                text_output = f"Sorry I met an error: {e}"
                seg_image = None
                instruction = ""

            chat_history.append(text_output, image_path=seg_image, role="expert")
            if instruction:
                chat_history.append(instruction, role="expert")
                outputs = self.generate_response(
                    messages=self.squash_expert_messages_into_user(chat_history.messages),
                    max_tokens=sv.max_tokens,
                    temperature=sv.temperature,
                    top_p=sv.top_p,
                    system_prompt=sv.sys_prompt,
                )
                chat_history.append(outputs, role="assistant")

        new_sv = new_session_variables(
            # Keep these parameters accross one conversation
            sys_prompt=sv.sys_prompt,
            sys_msg=sv.sys_msg,
            use_model_cards=sv.use_model_cards,
            temp_working_dir=sv.temp_working_dir,
            max_tokens=sv.max_tokens,
            temperature=sv.temperature,
            top_p=sv.top_p,
            interactive=True,
            sys_msgs_to_hide=sv.sys_msgs_to_hide,
            backup={"image_url": sv.image_url, "slice_index": sv.slice_index},
            img_urls_or_paths=sv.img_urls_or_paths,
        )

        logger.debug(f"<== Process the image and return the result")
        return (new_sv, chat_history)
