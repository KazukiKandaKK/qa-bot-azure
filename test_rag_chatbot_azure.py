import unittest
import sys
import os
import types
import importlib.machinery
from unittest.mock import Mock, patch

# Provide a minimal openai module if not installed
if 'openai' not in sys.modules:
    openai_module = types.ModuleType('openai')
    openai_module.__spec__ = importlib.machinery.ModuleSpec('openai', loader=None)
    class AzureOpenAI:  # dummy placeholder
        def __init__(self, *args, **kwargs):
            pass
    openai_module.AzureOpenAI = AzureOpenAI
    sys.modules['openai'] = openai_module

# Add repo root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_qa_chatbot_azure import KnowledgeBaseManager, QAChatbotAzure


class TestQAChatbotAzure(unittest.TestCase):
    """QAChatbotAzure class tests"""

    def setUp(self):
        # Mock knowledge base
        self.mock_kb = Mock(spec=KnowledgeBaseManager)
        self.mock_kb.search.return_value = ["これはテストコンテキストです。"]
        # Instance without Azure client
        self.chatbot_no_client = QAChatbotAzure(self.mock_kb)

    def test_build_prompt(self):
        context = ["情報1", "情報2"]
        question = "質問は？"
        prompt = self.chatbot_no_client._build_prompt(context, question)
        self.assertIn(context[0], prompt)
        self.assertIn(question, prompt)

    def test_simple_answer_with_context(self):
        context = ["経費の上限は月額5万円です。"]
        question = "経費の上限は？"
        answer = self.chatbot_no_client._simple_answer(context, question)
        self.assertIsInstance(answer, str)
        self.assertNotEqual(answer, "申し訳ありませんが、関連する情報が見つかりませんでした。")

    def test_simple_answer_no_context(self):
        answer = self.chatbot_no_client._simple_answer([], "?" )
        self.assertEqual(answer, "申し訳ありませんが、関連する情報が見つかりませんでした。")

    def test_answer_without_client(self):
        with patch.object(self.chatbot_no_client, '_simple_answer', return_value="simple") as mock_simple:
            result = self.chatbot_no_client.answer("質問")
            self.assertEqual(result, "simple")
            mock_simple.assert_called_once()

    def test_answer_with_client_success(self):
        with patch('rag_qa_chatbot_azure.AzureOpenAI') as mock_cls:
            mock_client = Mock()
            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = "azure response"
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_cls.return_value = mock_client

            chatbot = QAChatbotAzure(self.mock_kb, azure_endpoint="https://test", api_key="key")
            result = chatbot.answer("質問")
            self.assertEqual(result, "azure response")
            mock_client.chat.completions.create.assert_called_once()

    def test_answer_with_client_failure(self):
        with patch('rag_qa_chatbot_azure.AzureOpenAI') as mock_cls:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("error")
            mock_cls.return_value = mock_client

            chatbot = QAChatbotAzure(self.mock_kb, azure_endpoint="https://test", api_key="key")
            with patch.object(chatbot, '_simple_answer', return_value="fallback") as mock_simple:
                result = chatbot.answer("質問")
                self.assertEqual(result, "fallback")
                mock_simple.assert_called_once()

    def test_test_connection_no_client(self):
        self.assertFalse(self.chatbot_no_client.test_connection())

    def test_test_connection_success(self):
        with patch('rag_qa_chatbot_azure.AzureOpenAI') as mock_cls:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = Mock()
            mock_cls.return_value = mock_client

            chatbot = QAChatbotAzure(self.mock_kb, azure_endpoint="https://test", api_key="key")
            self.assertTrue(chatbot.test_connection())
            mock_client.chat.completions.create.assert_called_once()

    def test_test_connection_failure(self):
        with patch('rag_qa_chatbot_azure.AzureOpenAI') as mock_cls:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("fail")
            mock_cls.return_value = mock_client

            chatbot = QAChatbotAzure(self.mock_kb, azure_endpoint="https://test", api_key="key")
            self.assertFalse(chatbot.test_connection())
            mock_client.chat.completions.create.assert_called_once()


if __name__ == '__main__':
    unittest.main()
