from typing import List, Literal
import dspy
from pydantic import BaseModel


# ---------- 1. 统一 Relation Schema ---------- #
class BaseQuad(BaseModel):
    """
    统一的四元组结构：
    - time      : 事件发生的时间（可留空""，表示未知或不适用）
    - subject   : 事件的主体 / 执行者
    - location  : 事件发生地（可留空""）
    - predicate : 动作或关系（谓语/事件）
    """
    time: str
    subject: str
    location: str
    predicate: str


# ---------- 2. Signature 生成器 ---------- #
def _quad_sig(Quad: BaseModel, is_conversation: bool, context: str = "") -> dspy.Signature:
    """
    根据场景返回不同的 Signature：
      • 普通文本：允许 time/location 为空串
      • 对话文本：要求四要素全部来自给定实体列表
    """
    if not is_conversation:
        class ExtractTextQuads(dspy.Signature):
            __doc__ = f"""从源文本中提取"时间-主体-地点-事件"四元组。
            subject 必须出现在实体列表；time 可以为空。
            请不要出现“此时“这样不准确的描述，描述需要最大限度的精确
            请尽量完整、准确，忠实于原始文本。{context}"""
            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            quads: list[Quad] = dspy.OutputField(desc="四元组列表。")
        return ExtractTextQuads
    else:
        class ExtractConvQuads(dspy.Signature):
            __doc__ = f"""从对话中提取时间-主体-地点-事件四元组。
            subject 必须出现在实体列表；time 可以为空。
            四个字段的值都必须严格匹配实体列表中的内容。
            请尽量完整、准确，忠实于原始内容。{context}"""
            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            quads: list[Quad] = dspy.OutputField(desc="严格匹配实体列表的四元组列表。")
        return ExtractConvQuads


# ---------- 3. 回退方案 (无 Literal 限制) ---------- #
def _fallback_sig(entities: list[str], is_conversation: bool, context: str = ""):
    """发生类型校验错误时降级使用的 Quad 与 Signature"""
    ents = "\n- ".join(entities)

    class Quad(BaseModel):
        __doc__ = f"""四元组结构。subject 必须是以下实体之一：\n- {ents}"""
        time: str
        subject: str
        location: str
        predicate: str

    return Quad, _quad_sig(Quad, is_conversation, context)


# ---------- 4. 对外主函数 ---------- #
def get_quads(
    dspy: dspy.dspy,
    entities: list[str],
    is_conversation: bool = False,
    input_data: str = ""
) -> List[tuple[str, str, str, str]]:
    """
    抽取并返回 (time, subject, location, predicate) 四元组列表。
    """

    # 用 Literal 对 subject/location 做严格验证；time 仍允许任意字符串
    class Quad(BaseModel):
        time: str
        subject: Literal[tuple(entities)]
        location: str
        predicate: str

    ExtractQuads = _quad_sig(Quad, is_conversation, input_data)

    try:
        extract = dspy.Predict(ExtractQuads)
        res = extract(source_text=input_data, entities=entities)
        return [(q.time, q.subject, q.location, q.predicate) for q in res.quads]

    except Exception:  # 类型不匹配时自动降级
        Quad, ExtractQuads = _fallback_sig(entities, is_conversation, input_data)
        extract = dspy.Predict(ExtractQuads)
        res = extract(source_text=input_data, entities=entities)

        # ---------- 5. 再次校正：确保 subject/location 在实体列表 ---------- #
        class FixQuads(dspy.Signature):
            """修复四元组，使 subject 和 location 严格落在实体列表内。"""
            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            quads: list[Quad] = dspy.InputField()
            fixed_quads: list[Quad] = dspy.OutputField()

        fixer = dspy.ChainOfThought(FixQuads)
        fixed = fixer(source_text=input_data, entities=entities, quads=res.quads)

        good_quads = [
            q for q in fixed.fixed_quads
            if q.subject in entities and q.location in entities
        ]
        return [(q.time, q.subject, q.location, q.predicate) for q in good_quads]
