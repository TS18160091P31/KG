from typing import List, Literal
import dspy
from pydantic import BaseModel

def extraction_sig(Relation: BaseModel, is_conversation: bool, context: str = "") -> dspy.Signature:
  if not is_conversation:
    
    class ExtractTextRelations(dspy.Signature):
      __doc__ = f"""从源文本中提取主语-谓语-宾语三元组。
      主语和宾语必须来自提供的实体列表，这些实体已从相同的文本中提取。
      这是一个信息抽取任务，请尽可能详尽、准确，并忠实于原始文本。
       请为每个三元组附上其在原始文本中真实存在的 sentence 字段（即原文中包含该三元组的完整句子），不得构造，必须出现在 source_text 中。
         {context}"""
            
      source_text: str = dspy.InputField()
      entities: list[str] = dspy.InputField()
      relations: list[Relation] = dspy.OutputField(desc="主谓宾三元组组成的列表。请尽可能完整。")

    return ExtractTextRelations
  else:
    class ExtractConversationRelations(dspy.Signature):
      __doc__ = f"""从对话中提取主语-谓语-宾语三元组，包括以下类型：
      1. 对话中讨论概念之间的关系；
      2. 说话人与概念之间的关系（例如：用户询问了 X）；
      3. 说话人之间的关系（例如：助手回复了用户）。
      4. 关系可以使用下列内容中的表达：隶属,共事,归属,研制,搭载,合作,敌对,等同,部署
      主语和宾语必须出现在给定实体列表中，该实体列表已从相同文本中抽取。
      这是一个信息抽取任务，请尽量完整、准确，并忠于原始内容。
      请为每个三元组附上其在原始文本中真实存在的 sentence 字段（即原文中包含该三元组的完整句子），不得构造，必须出现在 source_text 中。 {context}"""
            
      source_text: str = dspy.InputField()
      entities: list[str] = dspy.InputField()
      relations: list[Relation] = dspy.OutputField(desc="主谓宾三元组组成的列表，主语和宾语必须严格匹配实体列表中的内容。请尽可能完整。")
      
    return ExtractConversationRelations
        

def fallback_extraction_sig(entities, is_conversation, context: str = "") -> dspy.Signature:
  """This fallback extraction does not strictly type the subject and object strings."""
  
  entities_str = "\n- ".join(entities)

  class Relation(BaseModel):
    __doc__ = f"""知识图谱中的主谓宾三元组结构。主语和宾语必须是以下实体之一： {entities_str}"""
    
    subject: str
    predicate: str
    object: str
    sentence: str  # 原文中该三元组所在的真实句子
    
  return Relation, extraction_sig(Relation, is_conversation, context)
  

def get_relations(dspy, input_data: str, entities: list[str], is_conversation: bool = False, context: str = "") -> List[str]:

  class Relation(BaseModel):
    """Knowledge graph subject-predicate-object tuple."""
    subject: Literal[tuple(entities)]
    predicate: str
    object: Literal[tuple(entities)]
  
  ExtractRelations = extraction_sig(Relation, is_conversation, context)
  
  try:
    
    extract = dspy.Predict(ExtractRelations)
    result = extract(source_text=input_data, entities=entities)
    return [(r.subject, r.predicate, r.object, r.sentence) for r in result.relations]
  
  except Exception as e:
    Relation, ExtractRelations = fallback_extraction_sig(entities, is_conversation, context)
    extract = dspy.Predict(ExtractRelations)
    result = extract(source_text=input_data, entities=entities)
    
    class FixedRelations(dspy.Signature):
      """修复关系三元组，确保每一条关系的主语和宾语都能在实体列表中精确匹配。
      谓语保持不变。每条关系的含义必须忠于原始文本。
      如果无法保持与原文一致的语义，则不要返回该关系。"""
            
      source_text: str = dspy.InputField()
      entities: list[str] = dspy.InputField()
      relations: list[Relation] = dspy.InputField()
      fixed_relations: list[Relation] = dspy.OutputField()
    
    fix = dspy.ChainOfThought(FixedRelations)
      
    fix_res = fix(source_text=input_data, entities=entities, relations=result.relations)
      
    good_relations = []
    for rel in fix_res.fixed_relations:
      if rel.subject in entities and rel.object in entities:
        good_relations.append(rel)
    return [(r.subject, r.predicate, r.object, r.sentence) for r in good_relations]