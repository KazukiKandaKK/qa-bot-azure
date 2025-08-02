#!/usr/bin/env python3
"""
Azure OpenAI版RAGチャットボットのデモンストレーション
実際のAzure設定例を含む
"""

import os
from rag_qa_chatbot_azure import KnowledgeBaseManager, QAChatbotAzure, setup_azure_config


def run_azure_demo_with_mock():
    """Azureクライアントをモックして動作確認"""
    print("="*60)
    print("🤖 Azure OpenAI版RAGチャットボット モックデモ")
    print("="*60)
    
    # ダミードキュメント
    documents = [
        "経費精算規定: 交通費は月額50,000円まで、接待費は1回10,000円まで申請可能です。",
        "有給休暇制度: 入社6ヶ月後に年10日付与、2年間有効です。",
        "IT サポート: システム障害時は内線1234番へ連絡してください。"
    ]
    
    # ナレッジベース構築
    print("📚 ナレッジベースを構築中...")
    kb_manager = KnowledgeBaseManager()
    kb_manager.build_index(documents)

    # Azure チャットボット初期化（設定なしでフォールバック）
    print("\n🤖 Azure チャットボットを初期化中...")
    chatbot = QAChatbotAzure(knowledge_base=kb_manager, company_name="株式会社デモ")
    
    # デモ質問
    questions = [
        "経費精算の上限を教えてください",
        "有給休暇はいつから取得できますか？",
        "システムにトラブルがあった時の連絡先は？"
    ]
    
    print("\n✨ デモンストレーション開始")
    for q in questions:
        print(f"\n📝 質問: {q}")
        answer = chatbot.answer(q)
        print(f"🤖 回答: {answer}")


def show_azure_setup_example():
    """Azure設定例の表示"""
    print("\n" + "="*60)
    print("🔧 Azure OpenAI 実際の設定例")
    print("="*60)
    
    print("\n1️⃣ 環境変数での設定例:")
    print("```bash")
    print("export AZURE_OPENAI_ENDPOINT='https://your-openai-resource.openai.azure.com/'")
    print("export AZURE_OPENAI_API_KEY='your-32-character-api-key'")
    print("```")
    
    print("\n2️⃣ Pythonコードでの設定例:")
    print("```python")
    print("from rag_qa_chatbot_azure import KnowledgeBaseManager, QAChatbotAzure")
    print()
    print("# ナレッジベース構築")
    print("kb_manager = KnowledgeBaseManager()")
    print("kb_manager.build_index(your_documents)")
    print()
    print("# Azure OpenAI チャットボット")
    print("chatbot = QAChatbotAzure(")
    print("    knowledge_base=kb_manager,")
    print("    azure_endpoint='https://your-openai-resource.openai.azure.com/',")
    print("    api_key='your-api-key',")
    print("    deployment_name='gpt-35-turbo',  # または 'gpt-4'")
    print("    api_version='2024-02-15-preview',")
    print("    company_name='your-company'")
    print(")")
    print()
    print("# 質問と回答")
    print("answer = chatbot.answer('あなたの質問')")
    print("print(answer)")
    print("```")
    
    print("\n3️⃣ Azure リソースで必要な設定:")
    print("• Azure OpenAI Service リソースの作成")
    print("• GPT-3.5-turbo または GPT-4 モデルのデプロイ") 
    print("• APIキーの取得（リソース > Keys and Endpoint）")
    print("• エンドポイントURLの確認")
    
    print("\n4️⃣ 推奨デプロイメント設定:")
    print("• モデル: gpt-35-turbo-16k または gpt-4")
    print("• バージョン: 最新の安定版")
    print("• 容量: 用途に応じて調整（TPM: Tokens Per Minute）")


def create_env_template():
    """環境変数テンプレートファイルの作成"""
    env_content = """# Azure OpenAI 設定
# 以下の値を実際の設定に置き換えてください

# Azure OpenAI リソースのエンドポイント
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/

# Azure OpenAI APIキー
AZURE_OPENAI_API_KEY=your-32-character-api-key-here

# デプロイメント名（オプション、コードで指定も可能）
AZURE_OPENAI_DEPLOYMENT=gpt-35-turbo

# APIバージョン（オプション）
AZURE_OPENAI_API_VERSION=2024-02-15-preview
"""
    
    with open("/Users/kazuki-k/qa/.env.template", "w", encoding="utf-8") as f:
        f.write(env_content)
    
    print("📝 環境変数テンプレートファイルを作成しました: .env.template")
    print("実際の値に置き換えて .env にリネームしてご使用ください。")


if __name__ == "__main__":
    # モックデモ実行
    run_azure_demo_with_mock()
    
    # Azure設定例表示
    show_azure_setup_example()
    
    # 環境変数テンプレート作成
    create_env_template()
    
    print("\n" + "="*60)
    print("✅ Azure OpenAI版デモ完了")
    print("="*60)
    print("\n💡 次のステップ:")
    print("1. Azure OpenAI リソースを作成")
    print("2. 環境変数または直接パラメータで設定")
    print("3. python rag_qa_chatbot_azure.py で実行")