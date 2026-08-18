from langchain_core.messages.human import HumanMessage

from graph import build_chat_graph

graph, pool = build_chat_graph()

if __name__ == "__main__":
    try:
        print("==== LangGraph多轮对话，输入quit退出 ====")
        # 演示用固定用户 ID；thread_id 决定短期记忆（会话内），user_id 决定长期记忆（跨会话）
        user_id = "user_001"
        while True:
            user_text = input("\n你：")
            if user_text.lower() == "quit":
                break
            config = {
                "configurable": {
                    "thread_id": user_id,  # 短期记忆：同一会话恢复历史
                    "user_id": user_id,    # 长期记忆：按用户隔离档案
                }
            }
            result = graph.invoke({"input_str": user_text}, config=config)
            ai_msg = result["messages"][-1]
            print(f"AI：{ai_msg.content}")
        # 退出循环后再关闭连接池
    finally:
        pool.close()
        print("bye")
