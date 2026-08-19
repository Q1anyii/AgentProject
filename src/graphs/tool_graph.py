from langchain_core.tools import tool


@tool(description="总结对话工具", parse_docstring=True)
def summarize():
    """
    该工具用于总结上下文，压缩当前会话

    Args:

    Returns:


    """


def build_tool_graph(self):
    return