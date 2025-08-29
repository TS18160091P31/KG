from typing import List,Dict
import dspy

class EventQuestionGenerator(dspy.Signature):
    """
    你是时政/军事分析专家，善于针对具体事件进行分析：根据输入的事件内容和分析维度，提出针对性问题。
    
    要求：
    - 每个维度至少提出num个具体专业的问题；
    - 问题必须与文章内容和维度高度相关，体现专业性；
    - 返回问题列表{dimension: [questions]}即可。
    """
    num:int=dspy.InputField(desc="至少提出的问题数目")
    event_article: str = dspy.InputField(desc="事件的简要内容")
    dimension: str = dspy.InputField(desc="分析的维度")
    questions: Dict[str, List[str]] = dspy.OutputField(desc="提出的具体分析问题列表,格式为 {dimension: [questions]}")


def generate_analysis_questions(dspy: dspy.dspy, event_article: str, dimension: str,num=5) -> List[str]:
    """
    输入：事件文章与分析维度
    输出：维度对应的问题列表
    """
    generator = dspy.Predict(EventQuestionGenerator)
    result = generator(event_article=event_article, dimension=dimension,num=num)
    return result.questions
