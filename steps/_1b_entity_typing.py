# src/kg_gen/steps/_1b_entity_typing.py
from typing import List
import dspy

class EntityWithType(dspy.Signature):
    """
    对输入实体进行分类，类别包括：人物、组织机构、国家地区、地点、设备、其他。
    要求：
    - 返回列表，列表中每项包含 name 和 type 字段；
    - 先合并同义/包含关系实体（如 “福特号” 与 “福特号航母打击群” → 保留前者）
    - type 只允许是以下类别之一："人物", "组织机构", "国家地区", "地点", "设备", "其他"，如果为其他则删除该项。
    - 尽可能依据上下文判断类型，而不是字面。
    """
    entity_list: list[str] = dspy.InputField(desc="实体字符串列表")
    context: str = dspy.InputField(desc="实体出现的原始文本上下文")
    typed_entities: list[dict] = dspy.OutputField(desc="每个实体及其类型，如 {'name': 'XXX', 'type': '设备'}")

def classify_entities(dspy: dspy.dspy, entity_list: List[str], context: str) -> List[dict]:
    """
    输入实体列表和原始上下文文本，返回带类型的实体结构。
    """
    classify = dspy.Predict(EntityWithType)
    result = classify(entity_list=entity_list, context=context)
    # print(dspy.inspect_history(n=1))
    return result.typed_entities
