#!/usr/bin/env python3
"""
RAGチャットボットの軽量デモ
実際のモデルダウンロードなしでデモンストレーション
"""

import sys
import os
from unittest.mock import Mock, patch
import numpy as np

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_demo():
    """軽量デモの実行"""
    print("="*60)
    print("🤖 RAG型社内QAチャットボット 軽量デモ")
    print("="*60)
    
    # モックを使用してシステムを初期化
    with patch('rag_qa_chatbot.SentenceTransformer') as mock_transformer, \
         patch('rag_qa_chatbot.faiss') as mock_faiss:
        
        print("📚 ナレッジベースを構築中...")
        
        # モック設定
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(10, 384).astype('float32')
        mock_transformer.return_value = mock_model
        
        mock_index = Mock()
        # 検索結果を質問に応じて変える
        def mock_search(query_embedding, k):
            # 質問の内容に応じて異なるインデックスを返す
            if hasattr(mock_search, 'call_count'):
                mock_search.call_count += 1
            else:
                mock_search.call_count = 0
            
            # 質問ごとに異なる結果をシミュレート
            results = [
                ([0.95, 0.85, 0.75], [0, 3, 6]),  # 経費精算関連
                ([0.92, 0.82, 0.72], [1, 4, 7]),  # 有給休暇関連
                ([0.88, 0.78, 0.68], [2, 5, 8]),  # システム関連
                ([0.65, 0.55, 0.45], [0, 1, 2])   # 関連性低い
            ]
            
            idx = min(mock_search.call_count, len(results) - 1)
            scores, indices = results[idx]
            return np.array([scores]), np.array([indices])
        
        mock_index.search = mock_search
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = Mock()
        
        from rag_qa_chatbot import KnowledgeBaseManager, QAChatbot
        
        # ダミーデータ
        dummy_documents = [
            "経費精算ルール: 交通費は月額50,000円まで、接待費は1回10,000円まで、申請期限は1ヶ月以内です。",
            "有給休暇申請ガイド: 入社6ヶ月後に10日間付与、有効期限2年間、3営業日前までに申請が必要です。",
            "社内システム利用ガイド: 社員番号とパスワードでログイン、90日ごとにパスワード変更、トラブル時はIT部門（内線1234）へ連絡。",
            "勤怠管理システム: 出退勤時刻の記録、有給申請、残業申請が可能です。",
            "セキュリティポリシー: パスワード管理、情報漏洩防止、外部デバイス利用制限について。",
            "福利厚生制度: 健康保険、厚生年金、退職金制度、社員食堂、図書購入補助について。",
            "出張規定: 出張申請手続き、宿泊費上限、交通費精算方法について。",
            "研修制度: 新人研修、スキルアップ研修、資格取得支援制度について。",
            "評価制度: 人事評価基準、昇進昇格制度、賞与査定について。",
            "緊急時対応: 災害時の避難経路、緊急連絡先、安否確認システムについて。"
        ]
        
        # ナレッジベース構築
        kb_manager = KnowledgeBaseManager()
        kb_manager.build_index(dummy_documents)
        
        # 検索結果を実際のドキュメントチャンクにマッピング
        def custom_search(query, top_k=3):
            if "経費" in query:
                return [dummy_documents[0], dummy_documents[6]]
            elif "有給" in query:
                return [dummy_documents[1]]
            elif "システム" in query or "ログイン" in query:
                return [dummy_documents[2], dummy_documents[3]]
            elif "会社" in query and "創立" in query:
                return []  # 情報なし
            else:
                return [dummy_documents[0]]  # デフォルト
        
        kb_manager.search = custom_search
        
        print("🤖 チャットボットを初期化中...")
        
        # チャットボット初期化（LLMなし）
        with patch('rag_qa_chatbot.AutoTokenizer'), \
             patch('rag_qa_chatbot.AutoModelForCausalLM'), \
             patch('rag_qa_chatbot.pipeline'):
            
            chatbot = QAChatbot(kb_manager)
            chatbot.generator = None  # シンプル回答を使用
            
            print("\n✨ デモンストレーション開始\n")
            
            # テスト質問
            test_questions = [
                "経費精算の上限はいくらですか？",
                "有給休暇はいつから取得できますか？", 
                "システムにログインできない場合はどうすればいいですか？",
                "会社の創立年はいつですか？"
            ]
            
            for i, question in enumerate(test_questions, 1):
                print(f"【質問 {i}】 {question}")
                print("-" * 50)
                
                try:
                    answer = chatbot.answer(question)
                    print(f"🤖 回答: {answer}")
                except Exception as e:
                    print(f"❌ エラー: {e}")
                
                print("\n")
            
            print("="*60)
            print("✅ デモンストレーション完了")
            print("="*60)
            
            print("\n📊 システム性能情報:")
            print(f"・総ドキュメント数: {len(dummy_documents)}")
            print(f"・総チャンク数: {len(kb_manager.chunks) if hasattr(kb_manager, 'chunks') else 'N/A'}")
            print("・ベクトル検索: FAISS (CPU版)")
            print("・回答生成: テンプレートベース")
            
            print("\n🔧 実際の運用時の推奨設定:")
            print("・Embeddingモデル: intfloat/multilingual-e5-small")
            print("・生成モデル: OpenAI GPT-4 または Claude")
            print("・ベクトルDB: FAISS、Chroma、または Pinecone")
            print("・チャンクサイズ: 300-500文字")
            print("・検索件数: 3-5件")

if __name__ == "__main__":
    run_demo()