import os
import json

from pathlib import Path
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Dict, Union
from openai import OpenAI
from sklearn.model_selection import train_test_split

from src.runs import AbstractRunStep
from src.utils.helper import load_yaml, unique_folder_path

import logging


class LLMStrategy(ABC):
    # Strategy interface for LLM Provider
    @abstractmethod
    def invoke(self, prompt: str):
        pass


class OpenAIStrategy(LLMStrategy):
    def __init__(self, config: Dict):
        """
        :param config: dict of config parameters.
        the accepted keys in config are:
        - open_ai_llm: dict of OpenAI llm parameters.
            - model: str, the model name.
        """

        self._client = OpenAI()
        self._model = config.get("open_ai_llm").get("model")

    def invoke(self, prompt: str):
        response = self._client.responses.create(
            model=self._model,
            input=prompt
        )
        return response.output_text


class PromptStrategy(ABC):
    _DEFAULT_PROMPT_TEMPLATE_FOLDER = Path(__file__).parent.parent.parent / "data" / "prompt_templates"
    _DEFAULT_DATASET_PATH = Path("../../data/datasets")
    _DEFAULT_SAVE_RESULT_PATH = Path("../../data/prompt_data_response")

    def __init__(self, prompt_template_name: str, llm: LLMStrategy, dataset_path: Union[str, Path]):
        self._prompt_template_name = prompt_template_name
        self._llm_provider = llm
        self._prompt_cache = PromptStrategy._load_prompt(self._prompt_template_name)
        self._dataset_path = dataset_path

    @property
    def llm_provider(self) -> LLMStrategy:
        return self._llm_provider

    @llm_provider.setter
    def llm_provider(self, llm: LLMStrategy) -> None:
        self._llm_provider = llm

    @classmethod
    @abstractmethod
    def _load_prompt(cls, prompt_template_name: str) -> Dict:
        dir_path = os.path.join(cls._DEFAULT_PROMPT_TEMPLATE_FOLDER, prompt_template_name + ".yaml")
        return load_yaml(dir_path)

    @abstractmethod
    def invoke(self, **kwargs):
        pass

    @abstractmethod
    def load_dataset_iterator(self, **kwargs):
        pass

    @abstractmethod
    def save_result(self, **kwargs):
        pass


class ZsCoTGossipCopRolePlayStrategy(PromptStrategy):
    # Zero shot CoT for Gossip Cop dataset

    _PROMPT_TEMPLATE_NAME = "role_play_prompt"
    _DATASET_NAME = "gossip_cop"

    def __init__(self, llm: LLMStrategy, dataset_name: str= _DATASET_NAME):
        """
        .. note::
            This invoker class cannot use parallel.
        :param llm:
        :param dataset_name:
        """
        dataset_path = PromptStrategy._DEFAULT_DATASET_PATH / dataset_name

        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset {dataset_name} not found")

        super().__init__(ZsCoTGossipCopRolePlayStrategy._PROMPT_TEMPLATE_NAME, llm, dataset_path)
        self._save_rs_path = unique_folder_path(
            PromptStrategy._DEFAULT_SAVE_RESULT_PATH, "ZsCoTGossipCopRolePlay")

    @classmethod
    def _load_prompt(cls, prompt_template_name: str) -> Dict:
        return super()._load_prompt(prompt_template_name)

    def invoke(self, **kwargs):

        input_text = kwargs.get("input_text")
        if not input_text:
            raise ValueError("input_text must be provided")

        the_good_prompt = self._prompt_cache.get("the_good").replace("{@input@}", input_text)

        logging.warning(f"Good: {the_good_prompt}")
        the_bad_prompt = self._prompt_cache.get("the_bad")

        the_ugly_prompt = self._prompt_cache.get("the_ugly").replace("{@input@}", input_text)
        logging.warning(f"Ugly: {the_bad_prompt}")
        the_good_explain = self.llm_provider.invoke(the_good_prompt)

        the_bad_prompt = the_bad_prompt.replace("{@input@}", input_text).replace(
            "{@the_good_explain@}", the_good_explain
        )
        logging.warning(f"Bad: {the_bad_prompt}")
        the_bad_explain = self.llm_provider.invoke(the_bad_prompt)

        the_ugly_explain = self.llm_provider.invoke(the_ugly_prompt)

        return the_good_explain, the_bad_explain, the_ugly_explain

    def load_dataset_iterator(self, **kwargs):
        """
        Load dataset
        :param kwargs: Configuration parameters.
        :return: (list) List of data points.

        The accepted keys in **kwargs include:
        - data_split_mode (str): the split dataset in [train, val, test]

        """
        data_split = kwargs.get("data_split_mode")
        if not data_split:
            raise ValueError("data_split_mode must be provided")

        dataset_file = data_split + ".json"

        dataset_path = self._dataset_path / dataset_file
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_result(self, **kwargs):
        """
        save the result to file JSON.

        :param kwargs: configuration parameters.
            the accepted keys in **kwargs include:
            - response_dict (dict): the dictionary of responses.
            - mode (str): the mode of the response, train, val or test.
            - file_name (str): the name of the file.

        :return: (None)
        """

        if not kwargs.get("response_dict"):
            raise ValueError("response must be provided")

        if kwargs.get("mode") not in ["train", "val", "test"]:
            raise ValueError("mode must be train, val or test")

        mode = kwargs.get("mode")
        file_name = kwargs.get("file_name") + ".json"
        data_path = self._save_rs_path / mode
        explain = kwargs.get("response_dict")

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        with open(data_path / file_name, "w", encoding="utf-8") as f:
            json.dump(explain, f, indent=4, ensure_ascii=False)


class LLMInvokeTrainStep(AbstractRunStep):
    def __init__(self, prompt_process: PromptStrategy):
        self._prompt = prompt_process

    def run(self, **kwargs):
        """
        Run the LLM invoker in a train model pipeline. Invoke the LLM provider and save the result.
        :param kwargs: Configuration parameters.
        The accepted keys in **kwargs include:
            - fetch_sample_data (bool): whether to fetch sample data from the dataset.
            - fetch_sample_size (float): the size of sample data to fetch. If "fetch_sample_data" is True,
                "fetch_sample_size" must be provided.
        """
        # TODO meta data file in save folder
        # TODO turn data into numpy array

        if kwargs.get("fetch_sample_data") not in [True, False]:
            raise ValueError("fetch_sample_data must be provided")

        fetch_sample_flag = kwargs.get("fetch_sample_data")

        if kwargs.get("fetch_sample_data"):
            if not kwargs.get("fetch_sample_size"):
                raise ValueError("fetch_sample_size must be provided")

        fetch_sample_size = kwargs.get("fetch_sample_size")

        for data_split_mode in ["train", "val", "test" ]:

            print(f"Start interacting with external LLM with {data_split_mode} dataset.")
            origin_dataset = self._prompt.load_dataset_iterator(data_split_mode=data_split_mode)
            dataset = [[item['content'], item['label']] for item in origin_dataset]

            dataset_label = [item['label'] for item in origin_dataset]
            if fetch_sample_flag:
                dataset, _ = train_test_split(dataset, test_size=fetch_sample_size, stratify=dataset_label,
                                              random_state=42)
            for idx, item in enumerate(tqdm(dataset)):
                the_good_explain, the_bad_explain, the_ugly_explain = self._prompt.invoke(
                    input_text=item[0]
                )
                explain = {"news": item[0], "the_good": the_good_explain, "the_bad": the_bad_explain,
                           "the_ugly": the_ugly_explain, "label": item[1]}

                self._prompt.save_result(response_dict=explain, mode=data_split_mode, file_name=str(idx))