"""Agent 主逻辑"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from tools import search_movie, get_movie_details, recommend, get_user_history, coldstart_recommend

# 加载环境变量
load_dotenv()

# 用户记忆（简单版）
user_preferences = {}


def create_llm():
    """创建 LLM 实例"""
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL", "https://api.longcat.chat/openai/v1")
    model = os.getenv("MODEL", "LongCat-2.0")

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
        temperature=0.3,  # 降低温度，减少幻觉
        max_tokens=1024
    )


# 定义可用工具
tools = [search_movie, get_movie_details, recommend, get_user_history, coldstart_recommend]


def create_agent(user_id: int = 1):
    """创建 Agent"""
    llm = create_llm()
    llm_with_tools = llm.bind_tools(tools)

    # 系统提示词（严格限制幻觉）
    system_prompt = f"""你是一个智能电影推荐助手。

重要规则：
1. 只基于工具返回的真实数据回答，绝对不要编造电影名称或信息
2. 如果工具返回结果为空，如实告知用户
3. 推荐时显示电影标题和类型

用户类型：
- 老用户（在系统中有评分历史）：用 user_id 调用 recommend
- 新用户（无历史记录）：让用户输入喜欢的电影标题，用 search_movie 转 ID，再用 coldstart_recommend

你可以使用的工具：
- search_movie(title) - 按标题搜索电影
- get_movie_details(movie_id) - 获取电影详细信息
- recommend(user_id, genre=None) - 为老用户推荐（可按类型过滤）
- get_user_history(user_id) - 获取用户历史
- coldstart_recommend(liked_movie_ids) - 为新用户推荐

请用中文回答，简洁自然。只说工具返回的内容。"""

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


def chat(user_input: str, user_id: int = 1) -> str:
    """单次对话"""
    agent = create_agent(user_id)
    result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    return result["messages"][-1].content


def update_preference(user_id: int, key: str, value: str):
    """更新用户偏好记忆"""
    if user_id not in user_preferences:
        user_preferences[user_id] = {}
    user_preferences[user_id][key] = value


def get_preferences(user_id: int) -> dict:
    """获取用户偏好"""
    return user_preferences.get(user_id, {})


def interactive_chat():
    """交互式对话"""
    print("=" * 60)
    print("🎬 智能电影推荐助手")
    print("=" * 60)
    print("输入 'quit' 退出")
    print()

    user_id = 1  # 默认用户
    agent = create_agent(user_id)

    print(f"你好！我是你的电影推荐助手。请输入你的需求：\n")

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ["quit", "exit", "退出"]:
            print("再见！")
            break

        if not user_input:
            continue

        try:
            result = agent.invoke({
                "messages": [HumanMessage(content=user_input)]
            })
            response = result["messages"][-1].content
            print(f"\n助手: {response}\n")
        except Exception as e:
            print(f"\n错误: {str(e)}\n")


if __name__ == "__main__":
    interactive_chat()

