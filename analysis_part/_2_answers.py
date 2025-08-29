import typing 
import dspy

class EventAnswerGenerator(dspy.Signature):
    """
    你是时政/军事分析专家，擅长从多个角度深入分析具体事件。请根据输入的问题，结合事件的背景与相关资料，提供专业、客观且严谨的分析回答。

    要求：
    - 答案必须基于提供的文章内容，避免任何推测或捏造。
    - 每个回答需展现深入的分析与专业知识，确保与事件的各个方面紧密相关。
    - 所提供的回答应有条理且详细，能够清晰地回应提问，并提供足够的证据和背景信息支撑观点。
    - 请确保回答格式规范，简洁但不失深度。
    """
    event_article: str = dspy.InputField(desc="事件的完整文章内容")
    question: str = dspy.InputField(desc="分析问题")
    answer: str = dspy.OutputField(desc="问题的详细回答")

def generate_analysis_answer(dspy: dspy.dspy, event_article: str, question: str) -> str:
    """
    输入：事件文章与分析维度
    输出：回答
    """
    generator = dspy.Predict(EventAnswerGenerator)
    result = generator(event_article=event_article, question=question)
    return result.answer
