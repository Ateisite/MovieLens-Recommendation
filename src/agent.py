"""Agent 主逻辑"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from tools import search_movie, get_movie_details, recommend, get_user_history

# 加载环境变量
load_dotenv()


def create_llm():
    """创建 LLM 实例"""
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("MODEL", "deepseek-chat")

    if not api_key or api_key == "sk-your-deepseek-key":
        raise ValueError(
            "请配置 API Key！\n"
            "1. 复制 .env.example 为 .env\n"
            "2. 填入你的 DeepSeek/LongCat API Key"
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.7,
        max_tokens=1024
    )


# 定义可用工具
tools = [search_movie, get_movie_details, recommend, get_user_history]


def create_agent():
    """创建 Agent"""
    llm = create_llm()
    llm_with_tools = llm.bind_tools(tools)

    # 系统提示词
    system_prompt = """你是一个智能电影推荐助手。你可以使用以下工具：

1. search_movie(title) - 按标题搜索电影
2. get_movie_details(movie_id) - 获取电影详细信息
3. recommend(user_id) - 为用户生成个性化推荐
4. get_user_history(user_id) - 获取用户历史评分

工作流程：
- 用户询问推荐时，先确认 user_id，然后调用 recommend
- 用户搜索电影时，调用 search_movie
- 用户想了解电影详情时，调用 get_movie_details
- 生成推荐后，用简洁语言解释推荐理由

请用中文回答，语气友好自然。"""

    def agent_node(state: MessagesState):
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # 构建图
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("agent")

    # 条件边：如果有工具调用，转到 tools；否则结束
    def should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "__end__"

    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "__end__": "__end__"
    })
    graph.add_edge("tools", "agent")

    return graph.compile()


def chat(user_input: str, user_id: int = None):
    """单次对话"""
    agent = create_agent()

    # 构建输入
    if user_id:
        input_text = f"[用户ID: {user_id}] {user_input}"
    else:
        input_text = user_input

    result = agent.invoke({"messages": [HumanMessage(content=input_text)]})

    # 返回最后一条消息
    return result["messages"][-1].content


def interactive_chat():
    """交互式对话"""
    print("=" * 60)
    print("🎬 智能电影推荐助手")
    print("=" * 60)
    print("输入 'quit' 退出")
    print()

    agent = create_agent()

    # 获取用户 ID
    user_id = input("请输入你的用户 ID (1-6503): ").strip()
    if not user_id.isdigit():
        user_id = 1
    else:
        user_id = int(user_id)

    print(f"\n你好，用户 {user_id}！请输入你的需求：\n")

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ["quit", "exit", "退出"]:
            print("再见！")
            break

        if not user_input:
            continue

        try:
            result = agent.invoke({
                "messages": [HumanMessage(content=f"[用户ID: {user_id}] {user_input}")]
            })
            response = result["messages"][-1].content
            print(f"\n助手: {response}\n")
        except Exception as e:
            print(f"\n错误: {str(e)}\n")


if __name__ == "__main__":
    interactive_chat()
