from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from kg_pipeline import run_kg, run_structured_qa
import requests
import os
from typing import List, Tuple, Dict, Any, Optional, Union
from QA_part._1_text import get_relevant_answer, generate_embeddings, split_text_into_sentences, \
    split_text_into_paragraphs  # Add the new function import

app = FastAPI(title="KG Extraction Service",
              description="封装 kg_gen 流水线，提供实体属性 / 关系抽取服务,问答服务")
# 初始化
from config import settings
import dspy

lm = dspy.LM(
    model=settings.OLLAMA_MODEL,
    api_key=settings.API_KEY,
    api_base=settings.OLLAMA_BASE
)
dspy.configure(lm=lm)


class QARequest(BaseModel):
    question: str
    context: str


class QAResponseItem(BaseModel):
    answer: List[Any]  # ["结论", "原文", ["解释1", "解释2"]]
    confidence: Dict[str, Any]  # {"推断可信度": {...}, "信息来源可靠性": {...}}


class QAResponse(BaseModel):
    results: List[QAResponseItem]
    note: Optional[str] = None


#问答结构化抽取（结论/证据/推理 + 置信度）
@app.post("/structured_extract", response_model=QAResponse)
async def structured_extract(req: QARequest):
    try:
        result = run_structured_qa(question=req.question, context=req.context)
        # 情况：原文未提及
        if isinstance(result, list) and len(result) == 0:
            return {
                "results": [],
                "note": "原文未提及相关信息，无法作答。"
            }
        combined = []
        for item in result:
            combined.append({
                "answer": [
                    item.get("conclusion", ""),
                    item.get("evidence", ""),
                    item.get("reasoning", [])
                ],
                "confidence": item.get("confidence", {})
            })
        return {
            "results": combined,
            "note": None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class EntityOutput(BaseModel):
    type: str
    word: str
    word_c: str
    start: int
    end: int


class KGResponse(BaseModel):
    string: str
    entities: List[EntityOutput]
    envent_detail: List[Any]
    envent_entity_rel: List[List[Any]]
    relationships: List[List[Any]]
    envent_svos: List[Tuple[str, str, str]]
    res_re_infer: List[Any]
    triple_pair_new: List[Any]


class KGRequest(BaseModel):
    text: str


# 知识图谱要素抽取（实体、关系、SVO、推理结果等）
@app.post("/extract", response_model=KGResponse)
async def extract(text: str = Form(None), file: UploadFile = File(None)):
    if text:
        result = run_kg(text)
        return result
    elif file:
        try:
            content = (await file.read()).decode("utf-8")
            return run_kg(content)
        except UnicodeDecodeError:
            raise HTTPException(400, "文件必须为 UTF‑8 文本")
    else:
        raise HTTPException(status_code=400, detail="请提供文本或文件")


#ollama连接测试
@app.get("/ping_ollama")
def ping_ollama():
    base_url = os.getenv("KG_OLLAMA_BASE", "http://localhost:11434")
    try:
        r = requests.get(base_url, timeout=3)
        return {"status": "ok", "response": r.text}
    except Exception as e:
        return {"status": "fail", "error": str(e)}


# Define the request model for the relevant paragraphs endpoint
class RelevantContextRequest(BaseModel):
    question: str
    context: str


class RelevantContextResponse(BaseModel):
    relevant_context: list
    note: Optional[str] = None
# Define the endpoint to return only the relevant paragraphs or sentences


#返回相似的段落
@app.post("/get_relevant_paragraphs", response_model=RelevantContextResponse)
async def get_relevant_paragraphs(req: RelevantContextRequest):
    try:
        # Get the relevant context (paragraphs/sentences) based on the input question and context
        relevant_context = get_relevant_answer(req.context, req.question, top_k=20)

        # If no relevant context is found
        if not relevant_context:
            return {
                "relevant_context": [],
                "note": "原文未提及相关信息，无法作答。"
            }
        # Return the relevant context
        return {
            "relevant_context": relevant_context,
            "note": None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#段落+回答
@app.post("/combined_extract")
async def combined_extract(req: QARequest):
    try:
        # Step 1: 获取相关上下文（get_relevant_paragraphs 部分逻辑）
        relevant_context = get_relevant_answer(req.context, req.question, top_k=20)

        if not relevant_context:
            return {
                "relevant_context": [],
                "note": "原文未提及相关信息，无法作答。",
                "results": []
            }

        # Step 2: 用 relevant_context 拼成新的上下文传给 structured_extract
        merged_context = "\n".join(relevant_context)
        structured_result = run_structured_qa(question=req.question, context=merged_context)

        results = []
        for item in structured_result:
            results.append({
                "answer": [
                    item.get("conclusion", ""),
                    item.get("evidence", ""),
                    item.get("reasoning", [])
                ],
                "confidence": item.get("confidence", {})
            })

        return {
            "relevant_context": relevant_context,
            "note": None,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class EmbeddingRequest(BaseModel):
    text: str
    model: Optional[str] = "nomic-embed-text"


class EmbeddingResponse(BaseModel):
    text: str
    embedding: List[float]
    dimension: int


#获取句子向量
@app.post("/get_embedding", response_model=EmbeddingResponse)
def get_embedding(req: str = Form(None)):
    try:
        response = generate_embeddings(req)
        return {
            "text": req,
            "embedding": response["embedding"],
            "dimension": len(response["embedding"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama embedding error: {str(e)}")


class SplitRequest(BaseModel):
    text: str


class SplitResponse(BaseModel):
    sentences: List[str]
    count: int


#将文章分割为句子
@app.post("/split_sentences", response_model=SplitResponse)
def split_sentences(req: SplitRequest):
    try:
        sentences = split_text_into_sentences(req.text)
        return {
            "sentences": sentences,
            "count": len(sentences)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentence splitting error: {str(e)}")


#文章分割为段落
@app.post("/split_paragraphs", response_model=SplitResponse)
def split_paragraphs(req: SplitRequest):
    try:
        sentences = split_text_into_paragraphs(req.text)
        return {
            "sentences": sentences,
            "count": len(sentences)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentence splitting error: {str(e)}")


from analysis_part_new._1_question import generate_analysis_questions


class GenerateQuestionRequest(BaseModel):
    event_article: str  # 事件文章内容
    dimension: str  # 分析维度
    num: int = 5  # 至少提出的问题数


class GenerateQuestionResponse(BaseModel):
    questions: dict  # {维度: [问题列表]}


#生成问题
@app.post("/generate_questions", response_model=GenerateQuestionResponse)
async def generate_questions(req: GenerateQuestionRequest):
    try:

        questions = generate_analysis_questions(dspy=dspy, event_article=req.event_article, dimension=req.dimension,
                                                num=req.num)
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from analysis_part_new._1b_question import generate_detailed_questions


class GenerateDetailRequest(BaseModel):
    text: str  # 事件基本内容
    question: str  # 分析维度


class GenerateDetailResponse(BaseModel):
    detail_q: list  # {维度: [问题列表]}


#生成回答问题需要的信息
@app.post("/generate_details", response_model=GenerateDetailResponse)
async def generate_details(req: GenerateDetailRequest):
    try:
        detail_q = generate_detailed_questions(dspy=dspy, event_title=req.text, question=req.question)
        # print(f"  回答: {answer}")
        return {"detail_q": detail_q[req.question]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from analysis_part_new._2_answers import generate_analysis_answer


class AnswerQuestionRequest(BaseModel):
    event_article: str  # 事件文章内容
    question: str  # 分析问题


class AnswerQuestionResponse(BaseModel):
    answer: str  # 问题的详细回答
    confidence: str


@app.post("/answer_question", response_model=AnswerQuestionResponse)
async def answer_question(req: AnswerQuestionRequest):
    try:
        res = generate_analysis_answer(
            dspy=dspy,
            event_article=req.event_article,
            question=req.question
        )

        # 兼容不同返回类型
        if isinstance(res, dict):
            answer = res.get("answer") or res.get("text") or str(res)
            confidence = res.get("confidence") or res.get("score") or "未知"
        else:
            answer = str(res)
            confidence = "未知"

        # 返回与 response_model 完全一致的结构（字符串 + 字符串）
        return AnswerQuestionResponse(answer=answer, confidence=confidence)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from analysis_part_new._1c_rank import generate_rank


class RankRequest(BaseModel):
    main_question: str  # 事件文章内容
    sub_questions: Union[List[str], str]  # 分析问题


class RankItem(BaseModel):
    sub_question: str  # 子问题文本
    score: int  # 0~3 分
    rationale: str  # 打分理由


class RankResponse(BaseModel):
    answer: List[RankItem]  # 问题的详细回答


#回答问题
@app.post("/get_rank", response_model=RankResponse)
async def get_rank(req: RankRequest):
    try:
        answer = generate_rank(dspy=dspy, main_question=req.main_question, sub_questions=req.sub_questions)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================================
# 1) /entity_types —— 实体类型判定（过滤掉“其他”）
# =====================================================================================
from steps._1b_entity_typing import classify_entities
from steps._1_get_entities import get_entities


class EntityTypesRequest(BaseModel):
    context: str


class EntityTypeItem(BaseModel):
    # 按你规则保留 5 类（人物/组织机构/国家地区/地点/设备），自动丢弃“其他”
    name: str
    type: str


@app.post("/entity_types", response_model=List[EntityTypeItem])
async def entity_types(req: EntityTypesRequest):
    try:
        entities = get_entities(dspy=dspy, input_data=req.context, is_conversation=False)
        typed = classify_entities(dspy=dspy, entity_list=entities, context=req.context)
        return typed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================================
# 2) /entity_attrs —— 航母实体属性抽取（固有属性 + 最新官方状态）
# =====================================================================================
from steps.entity_attr import extract_entity_attributes


class EntityAttrsRequest(BaseModel):
    # 仅传入“具体航母”的实体名列表（类型应为“设备”）
    entity_list: List[str]
    context: str


class EntityAttrsItem(BaseModel):
    name: str
    attributes: List[str]


config_path = "config/entity_attr.yaml"


@app.post("/entity_attrs", response_model=List[EntityAttrsItem])
async def entity_attrs(req: EntityAttrsRequest):
    try:
        attrs = extract_entity_attributes(
            entity_list=req.entity_list,
            context=req.context,
            config_path=config_path
        )
        return attrs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================================
# 3) /relations —— 关系四元组抽取 (time, subject, location, predicate)
# =====================================================================================
from steps._2_get_relations import get_quads


class RelationsRequest(BaseModel):
    source_text: str
    entities: List[str]


class QuadItem(BaseModel):
    time: str
    subject: str
    location: str
    predicate: str


@app.post("/relations", response_model=List[QuadItem])
async def relations(req: RelationsRequest):
    try:
        quads = get_quads(
            dspy=dspy,
            input_data=req.source_text,
            entities=req.entities,
        )
        # get_quads 返回 List[tuple]，这里转成 dict 以匹配响应模型
        answer = [
            {"time": t, "subject": s, "location": loc, "predicate": p}
            for (t, s, loc, p) in quads
        ]
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    ##别名定位


from QA_part.find_alias import find_alias_positions
from steps._1d_entity_aliases import extract_aliases_llm


class AliasMatch(BaseModel):
    alias: str
    start: int
    end: int


class AliasRequest(BaseModel):
    text: str
    entities: List[str]


@app.post("/alias_positions", response_model=Dict[str, List[AliasMatch]])
async def relations(req: AliasRequest):
    entity_aliases = {}
    for ent in req.entities:
        aliases = extract_aliases_llm(dspy, entity_name=ent, context=req.text)
        entity_aliases[ent] = aliases
    entity_positions_raw = find_alias_positions(req.text, entity_aliases)
    return entity_positions_raw
# from fastapi.responses import StreamingResponse
# from pathlib import Path

# @app.get("/stream/{file_path:path}")
# async def stream_file(file_path: str):
#     file = Path("outputs") / file_path
#     if not file.exists():
#         return {"error": "File not found"}

#     def iterfile():
#         with open(file, "rb") as f:
#             while chunk := f.read(1024 * 1024):
#                 yield chunk

#     return StreamingResponse(
#         iterfile(),
#         media_type="application/octet-stream",
#         headers={"Content-Disposition": f"attachment; filename={file.name}"}
#     )
