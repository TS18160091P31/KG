import dspy

class EntityAliasExtraction(dspy.Signature):
    """
    任务：从原文中提取指定实体的所有别名（这些别名必须在原文中真实出现）。

    要求：
    - 返回所有与该实体指代相同的文本表达；
    - 别名必须完整匹配原文中出现的内容，不能是模型生成的；
    - 例如：实体“福特号”在原文中也叫“福特号航母打击群”、“CVN-78”等；
    - 输出为字符串数组。

    示例输出：
    ["福特号", "福特号航母打击群", "CVN-78"]
    """

    entity_name: str = dspy.InputField()
    context: str = dspy.InputField()
    aliases: list[str] = dspy.OutputField()
def extract_aliases_llm(dspy, entity_name: str, context: str) -> list[str]:
    alias_predictor = dspy.Predict(EntityAliasExtraction)
    result = alias_predictor(entity_name=entity_name, context=context)
    return result.aliases
