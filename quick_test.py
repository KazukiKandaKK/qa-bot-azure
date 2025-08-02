#!/usr/bin/env python3
"""
RAGチャットボットの軽量テスト
モデルのダウンロードなしでコア機能をテスト
"""

import sys
import os
from unittest.mock import Mock, patch
import numpy as np

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_document_splitting():
    """ドキュメント分割のテスト"""
    print("🔍 ドキュメント分割機能をテスト中...")
    
    # KnowledgeBaseManagerクラスをインポート（モデルなしで初期化）
    with patch('rag_qa_chatbot.SentenceTransformer') as mock_transformer:
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        from rag_qa_chatbot import KnowledgeBaseManager
        kb = KnowledgeBaseManager()
        
        test_doc = "これは最初の文です。これは二番目の文です。これは三番目の文です。長い文章をテストするための追加のテキストです。"
        chunks = kb._split_document(test_doc)
        
        print(f"✅ チャンク数: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"   チャンク{i+1}: {chunk[:50]}...")
        
        assert len(chunks) > 0, "チャンクが作成されていません"
        print("✅ ドキュメント分割テスト合格\n")

def test_prompt_building():
    """プロンプト構築のテスト"""
    print("🔍 プロンプト構築機能をテスト中...")
    
    with patch('rag_qa_chatbot.SentenceTransformer'), \
         patch('rag_qa_chatbot.AutoTokenizer'), \
         patch('rag_qa_chatbot.AutoModelForCausalLM'), \
         patch('rag_qa_chatbot.pipeline'):
        
        from rag_qa_chatbot import QAChatbot, KnowledgeBaseManager
        
        mock_kb = Mock(spec=KnowledgeBaseManager)
        chatbot = QAChatbot(mock_kb)
        
        context = ["経費精算の上限は月額50,000円です。", "申請は月末までに行ってください。"]
        question = "経費精算について教えてください"
        
        prompt = chatbot._build_prompt(context, question)
        
        print("✅ 生成されたプロンプト:")
        print(prompt[:200] + "...")
        
        # プロンプトの内容チェック
        assert "株式会社○○" in prompt, "会社名が含まれていません"
        assert question in prompt, "質問が含まれていません"
        assert context[0] in prompt, "コンテキストが含まれていません"
        
        print("✅ プロンプト構築テスト合格\n")

def test_simple_answer():
    """シンプル回答機能のテスト"""
    print("🔍 シンプル回答機能をテスト中...")
    
    with patch('rag_qa_chatbot.SentenceTransformer'), \
         patch('rag_qa_chatbot.AutoTokenizer'), \
         patch('rag_qa_chatbot.AutoModelForCausalLM'), \
         patch('rag_qa_chatbot.pipeline'):
        
        from rag_qa_chatbot import QAChatbot, KnowledgeBaseManager
        
        mock_kb = Mock(spec=KnowledgeBaseManager)
        chatbot = QAChatbot(mock_kb)
        chatbot.generator = None  # シンプル回答を強制
        
        # コンテキストありのテスト
        context = ["経費精算の上限は月額50,000円です。"]
        question = "経費精算の上限はいくらですか？"
        answer = chatbot._simple_answer(context, question)
        
        print(f"✅ 回答例: {answer}")
        assert len(answer) > 0, "回答が生成されていません"
        assert "申し訳ありません" not in answer, "エラーメッセージが返されています"
        
        # コンテキストなしのテスト
        empty_answer = chatbot._simple_answer([], question)
        print(f"✅ 情報なし時の回答: {empty_answer}")
        assert "申し訳ありません" in empty_answer, "適切なエラーメッセージが返されていません"
        
        print("✅ シンプル回答テスト合格\n")

def test_integration():
    """統合テスト"""
    print("🔍 統合テストを実行中...")
    
    with patch('rag_qa_chatbot.SentenceTransformer') as mock_transformer, \
         patch('rag_qa_chatbot.faiss') as mock_faiss:
        
        # モック設定
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(3, 384).astype('float32')
        mock_transformer.return_value = mock_model
        
        mock_index = Mock()
        mock_index.search.return_value = (np.array([[0.9, 0.8]]), np.array([[0, 1]]))
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = Mock()
        
        from rag_qa_chatbot import KnowledgeBaseManager, QAChatbot
        
        # システム初期化
        documents = [
            "経費精算の上限は月額50,000円です。申請は期限内に行ってください。",
            "有給休暇は入社6ヶ月後から取得可能です。年間10日間付与されます。"
        ]
        
        kb_manager = KnowledgeBaseManager()
        kb_manager.build_index(documents)
        
        # チャットボット初期化（LLMなし）
        with patch('rag_qa_chatbot.AutoTokenizer'), \
             patch('rag_qa_chatbot.AutoModelForCausalLM'), \
             patch('rag_qa_chatbot.pipeline'):
            
            chatbot = QAChatbot(kb_manager)
            chatbot.generator = None  # シンプル回答を使用
            
            # 検索結果をモック
            kb_manager.search = Mock(return_value=["経費精算の上限は月額50,000円です。"])
            
            # テスト質問
            questions = [
                "経費精算の上限はいくらですか？",
                "有給休暇について教えてください",
                "存在しない情報について教えてください"
            ]
            
            for question in questions:
                print(f"📝 質問: {question}")
                answer = chatbot.answer(question)
                print(f"🤖 回答: {answer[:100]}...")
                assert len(answer) > 0, f"質問'{question}'に対する回答が生成されませんでした"
                print()
        
        print("✅ 統合テスト合格\n")

def main():
    """メインテスト実行関数"""
    print("="*60)
    print("🚀 RAG QAチャットボット 軽量テスト開始")
    print("="*60)
    
    try:
        test_document_splitting()
        test_prompt_building()
        test_simple_answer()
        test_integration()
        
        print("="*60)
        print("🎉 すべてのテストが正常に完了しました！")
        print("="*60)
        
        print("\n📋 テスト結果サマリー:")
        print("✅ ドキュメント分割機能: 正常")
        print("✅ プロンプト構築機能: 正常")
        print("✅ シンプル回答機能: 正常")
        print("✅ 統合テスト: 正常")
        
        print("\n💡 次のステップ:")
        print("1. pip install -r requirements.txt でライブラリをインストール")
        print("2. python rag_qa_chatbot.py で実際のデモを実行")
        print("3. python test_rag_chatbot.py で完全なテストスイートを実行")
        
    except Exception as e:
        print(f"❌ テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()