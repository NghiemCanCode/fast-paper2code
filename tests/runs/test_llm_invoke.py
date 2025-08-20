import unittest
from pathlib import Path

# import external module
from dotenv import load_dotenv

from src.runs.llm_invoke import ZsCoTGossipCopRolePlayStrategy, OpenAIStrategy, LLMInvokeTrainStep

class ZsCoTGossipCopFakeEviStrategyTestCase(unittest.TestCase):
    def test_init_method(self):
        load_dotenv()

        config_dict = {
            "open_ai_llm":
                {"model":"gpt-4.1-nano-2025-04-14"}
        }
        llm_strategy = OpenAIStrategy(config_dict)
        prompt = ZsCoTGossipCopRolePlayStrategy(llm=llm_strategy)

        self.assertIsNotNone(prompt)
        self.assertTrue("the_good" in prompt._prompt_cache)
        self.assertTrue("the_bad" in prompt._prompt_cache)
        self.assertTrue("the_ugly" in prompt._prompt_cache)

    def test_invoke_method(self):
        load_dotenv()
        config_dict = {
            "open_ai_llm":
                {"model":"gpt-4.1-nano-2025-04-14"}
        }
        llm_strategy = OpenAIStrategy(config=config_dict)
        prompt = ZsCoTGossipCopRolePlayStrategy(llm=llm_strategy)
        rps_1, rps_2, rps_3 = prompt.invoke(input_text="Ca sĩ việt nam toàn nổi lên nhờ scandal")

        print("exp_1:", rps_1, "\n")
        print("exp_2:", rps_2, "\n")
        print("exp_3:", rps_3, "\n")

        self.assertIsNotNone(rps_1)
        self.assertIsNotNone(rps_2)
        self.assertIsNotNone(rps_3)

    def test_load_dataset_iterator_method(self):
        load_dotenv()
        config_dict = {
            "open_ai_llm":
                {"model": "gpt-4.1-nano-2025-04-14"}
        }
        llm_strategy = OpenAIStrategy(config=config_dict)
        prompt = ZsCoTGossipCopRolePlayStrategy(llm=llm_strategy)
        dataset_iterator = prompt.load_dataset_iterator(data_split_mode="test")
        self.assertIsInstance(dataset_iterator, list)
        self.assertEqual(dataset_iterator[0]["content"], "In February, Alicia Silverstone and Christopher Jarecki announced they were divorcing after 20 years together. The couple has a six - year - old son together, Bear Blu. ( Photo : Frazer Harrison, Getty Images ) Alicia Silverstone has filed for divorce from her husband of nearly 13 years, actor and musician Christopher Jarecki. The divorce papers were filed in Los Angeles County Superior Court on Friday, according to the Associated Press. The Clueless star, 41, had separated from Jarecki, 47, in February after more than 20 years together as a couple. At the time the couple said in a statement that ` ` they still deeply love and respect each other and remain very close friends.'' The papers state the couple will share custody of their 7 - year - old son, Bear Blue")

    def test_save_dataset_method(self):
        load_dotenv()

        config_dict = {
            "open_ai_llm":
                {"model": "gpt-4.1-nano-2025-04-14"}
        }
        llm_strategy = OpenAIStrategy(config=config_dict)
        prompt = ZsCoTGossipCopRolePlayStrategy(llm=llm_strategy)

        prompt.save_result(response="Test thôi làm gì căng", mode="test", file_name="000")
        filepath = Path("../../data/prompt_data_response/ZsCoTGossipCopFakeEvi/test/000.txt")
        self.assertTrue(filepath.exists())


class LLMInvokeTrainStepTestCase(unittest.TestCase):
    def test_run_method(self):

        load_dotenv()

        config_dict = {
            "open_ai_llm":
                {"model":"gpt-4.1-nano-2025-04-14"}
        }
        llm_strategy = OpenAIStrategy(config=config_dict)
        prompt = ZsCoTGossipCopRolePlayStrategy(llm=llm_strategy)

        llm_step = LLMInvokeTrainStep(prompt)

        llm_step.run(fetch_sample_data=False)

if __name__ == '__main__':
    unittest.main()
