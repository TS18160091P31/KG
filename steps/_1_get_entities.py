from typing import List
import dspy

# ------------------------------------------------------------
# 纯文本场景：实体抽取
# ------------------------------------------------------------
class TextEntities(dspy.Signature):
    """
    任务：从【源文本】中抽取关键实体，实体通常充当主语或宾语。

    要求：
    - 仅依据原文，准确、全面地列出实体；
    - 请务必“完整”且“精准”。

    如果一个实体包含国家、职务和人名，请将其拆分为三个独立的实体字段。
    例如，将‘罗马尼亚经济部长波格丹·伊万’拆分为：‘罗马尼亚’，‘经济部长’，‘波格丹·伊万’。

    输出格式要求：
    返回一个 JSON 对象，包含名为 "entities" 的字段，其值为字符串数组。例如：
    {
      "entities": ["福特号", "美国", "福特级航空母舰", "中东", "以色列", "弗吉尼亚州基地"]
    }
    """

    source_text: str = dspy.InputField(desc="待处理文本")
    entities: list[str] = dspy.OutputField(desc="完整的关键实体列表")

# ------------------------------------------------------------
# 对话场景：实体抽取
# ------------------------------------------------------------
class ConversationEntities(dspy.Signature):
    """
    任务：从【对话】中抽取关键实体，实体通常充当主语或宾语。

    说明：
    - 需同时考虑显式提及的实体与对话参与者；
    - 请务必“完整”且“精准”，忠实于对话内容。
如果一个实体包含国家、职务和人名，请将其拆分为三个独立的实体字段。例如，将‘罗马尼亚经济部长波格丹·伊万’拆分为：‘罗马尼亚’，‘经济部长’，‘波格丹·伊万’。    """
    source_text: str = dspy.InputField(desc="对话文本")
    entities: list[str] = dspy.OutputField(desc="完整的关键实体列表")

# ------------------------------------------------------------
# 对外函数
# ------------------------------------------------------------
def get_entities(dspy: dspy.dspy,
                 input_data: str,
                 is_conversation: bool = False) -> List[str]:
    """
    根据场景调用对应的实体抽取 Signature。
    """
    Extractor = ConversationEntities if is_conversation else TextEntities
    extract = dspy.Predict(Extractor)
    result = extract(source_text=input_data)
    print(dspy.inspect_history(n=1))
    return result.entities
