from typing import List, Dict 
import dspy

class RAGInformationNeedGenerator(dspy.Signature):
    """
    你是时政/军事分析专家助手，擅长将复杂问题转换为信息需求清单，供RAG系统使用。

    你的任务：
    - 根据事件背景（event_title）和用户问题（question），生成多个适用于语义检索的详细信息需求；
    - 每条信息需求应具体、完整，能够单独作为搜索句子或段落的查询内容；
    - 避免：重复项、太短的词组、含糊表达（如“背景”“影响”“发展情况”）；
    - 输出格式为：{question: [信息需求1, 信息需求2, ...]}；

    输入要求：
    - event_title 应为完整、详细的事件描述，建议包括时间、地点、主要参与方、基本经过；
    - 问题应为分析型问句，如“如何评价此次事件的政治影响？”、“此次行动对地区局势有何影响？”等；

    信息需求撰写建议：
    - 包含主题词（如中国、美国、军方、联合国、媒体等）
    - 包含动作或因果关系（如表态、部署、谴责、升级、回应）
    - 包含时间状态词（如事发后、同日、第二天、冲突升级期间）
    - 包含事件具体已知信息（如佩洛西访台,美进行军事演习） 

    示例输出：
    {
        "此次访问对地区安全局势的影响？": [
            "中国政府在佩洛西访台后的军事部署或演习活动",
            "美国国防部针对佩洛西访台采取的军事或外交行动",
            "台湾地区对佩洛西访问的安全接待安排与政治表态",
            "日本等周边国家对佩洛西访台的安保政策反应",
            "佩洛西访台后南海或台海周边是否出现新的军事摩擦"
        ]
    }
    """

    event_title: str = dspy.InputField(desc="事件的背景")
    question: str = dspy.InputField(desc="需要分析和回答的具体问题")
    info_needs: Dict[str, List[str]] = dspy.OutputField(desc="格式为{question: [详细信息需求1, 信息需求2, ...]}")

def generate_detailed_questions(dspy: dspy.dspy, event_title: str, question: str) -> List[str]:
    """
    输入：简要事件描述 + 用户提问
    输出：适合向量检索的多条细化信息需求（可输入RAG召回模块）
    """
    generator = dspy.Predict(RAGInformationNeedGenerator)
    result = generator(event_title=event_title, question=question)
    return result.info_needs
