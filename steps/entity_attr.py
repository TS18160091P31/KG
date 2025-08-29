import json
import dspy
from typing import List, Dict, TypedDict
import yaml

# 可选：你的实体结构（运行时校验为主，不强绑 Literal）
class Entity(TypedDict):
    name: str
    type: str  # 例如 "设备"

def _build_doc_from_cfg(cfg: dict) -> str:
    blocks = cfg.get("prompt_blocks", [])
    return "\n".join(blocks).strip()
from typing import Any

def _normalize_entities(raw: Any, default_type: str, whitelist: set[str]) -> list[dict]:
    """
    接受 list[str|dict] 或 str，统一成 [{'name':..., 'type':...}]
    - 如果项没有 type：补 default_type
    - 如果有 whitelist：最后再按 whitelist 过滤
    """
    out: list[dict] = []

    # 允许直接传单个字符串
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"entity_list 应为 list[str|dict] 或 str，实际是 {type(raw)}")

    for e in raw:
        if isinstance(e, str):
            out.append({"name": e, "type": default_type})
        elif isinstance(e, dict):
            name = e.get("name")
            if not name:
                raise ValueError(f"字典实体缺少 name 字段：{e}")
            etype = e.get("type") or default_type     # 关键：如果没有type就补默认
            out.append({"name": name, "type": etype})
        else:
            raise TypeError(f"不支持的实体项类型：{type(e)}（值：{e!r}）")

    # 运行时白名单过滤（如果配置了）
    if whitelist:
        out = [e for e in out if e["type"] in whitelist]
    return out

def _build_signature_from_cfg(cfg: dict) -> type[dspy.Signature]:
    """
    动态创建一个 dspy.Signature 子类，docstring 和字段描述都来自配置。
    """
    doc = _build_doc_from_cfg(cfg)
    descs = cfg.get("descriptions", {})
    fields = {
        "__doc__": doc,
        "entity_list": dspy.InputField(desc=descs.get("entity_list", "")),
        "context": dspy.InputField(desc=descs.get("context", "")),
        "entity_attributes": dspy.OutputField(desc=descs.get("entity_attributes", "")),
    }
    sig_name = cfg.get("signature_name", "EntityAttributeExtraction")
    return type(sig_name, (dspy.Signature,), fields)

def load_extractor_from_config(config_path: str):
    """
    从配置文件加载并构造一个可调用的抽取器函数：
      extractor(entity_list, context) -> List[Dict]
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    Sig = _build_signature_from_cfg(cfg)
    predictor = dspy.Predict(Sig)
    whitelist = set(cfg.get("entity_type_whitelist", []))
    blacklist = set(cfg.get("disallow_entity_types", []))
    default_type = cfg.get("default_entity_type") or (next(iter(whitelist), "设备"))

    def extractor(entity_list, context: str) -> list[dict]:
        entity_list = _normalize_entities(entity_list, default_type, whitelist)
        if blacklist:
            entity_list = [e for e in entity_list if e["type"] not in blacklist]
        result = predictor(entity_list=entity_list, context=context)
        return result.entity_attributes

    return extractor

# 兼容你原来的函数名（新增 config_path 参数）
def extract_entity_attributes(entity_list: List[Entity], context: str, config_path: str) -> List[Dict]:
    """
    用法与原先一致，但提示词来自 config_path。
    需先在外部完成 dspy.configure(lm=...)。
    """
    extractor = load_extractor_from_config(config_path)
    out=extractor(entity_list=entity_list, context=context)
    if isinstance(out, str):
        out = json.loads(out)
    if not isinstance(out, list):
        raise TypeError(f"模型输出应为 list，实际是 {type(out)}")
    
    return out
if __name__ == "__main__":
    lm = dspy.LM(
            "ollama_chat/deepseek-r1:14b",
            api_key="",
            api_base="http://localhost:11434"
        )
    dspy.configure(lm=lm)
    # entity_list = [
    # {"name": "福特号", "type": "设备"},
    # {"name": "某不相关", "type": "其他"},  # 会被过滤
    # ]
    entity_list=["福特号",{"name":"上海","type":"地点"},"弗吉尼亚州"]
    context = """2025-06-24，福特号启程驶离弗吉尼亚州，计划部署至地中海。
    其为福特级首舰，核动力，2017年正式服役，常驻诺福克。
    """

    out = extract_entity_attributes(entity_list, context, "config/entity_attr.yaml")
    if isinstance(out, str):
        out = json.loads(out)
    if not isinstance(out, list):
        raise TypeError(f"模型输出应为 list，实际是 {type(out)}")
    

    print(out)