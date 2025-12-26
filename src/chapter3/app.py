import os
import sys
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from tenacity import retry, wait_random_exponential, stop_after_attempt

# 引入必要的 OpenAI 异常类
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from tqdm import tqdm

load_dotenv()
# ==========================================
# 0. 基础设施层 (Infrastructure Layer)
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("System")


class AppConfig:
    def __init__(self):
        # 替换为你的 API Key
        self.api_key = os.getenv("MODELSCOPE_API_KEY")
        self.base_url = os.getenv(
            "MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1"
        )
        self.model_id = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
        self.max_retries = int(os.getenv("MAX_RETRIES", 3))
        self.retry_delay = int(os.getenv("RETRY_DELAY", 2))
        self._validate()

    def _validate(self):
        if not self.api_key:
            raise ValueError("Critical Config Error: API Key is missing.", self.api_key)


@dataclass
class McpMessage:
    sender: str
    content: str | Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    protocol_version: str = "1.0"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    def validate(self) -> bool:
        """检查消息核心字段是否完整"""
        if not self.sender or self.content is None:
            logger.error(f"MCP 校验失败：来自 {self.sender} 的内容为空")
            return False
        return True


class LLMService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def _print_stream(self, text: str):
        print(text, end="", flush=True)

    def chat_completion(
        self,
        system_prompt: str,
        user_content: str,
        enable_thinking: bool = True,
        json_mode: bool = True,
    ) -> str:
        logger.info(f"正在调用 LLM... (最大重试次数: {self.config.max_retries})")

        for attempt in range(1, self.config.max_retries + 1):
            try:
                return self._execute_call(
                    system_prompt, user_content, enable_thinking, json_mode
                )
            except (APIConnectionError, RateLimitError, APIError) as e:
                logger.warning(f"尝试 {attempt}/{self.config.max_retries} 失败: {e}")
                if attempt == self.config.max_retries:
                    raise e
                time.sleep(self.config.retry_delay)
            except Exception as e:
                logger.error(f"不可恢复的错误: {e}")
                raise e
        return ""

    def _execute_call(
        self,
        system_prompt: str,
        user_content: str,
        enable_thinking: bool,
        json_mode: bool,
    ) -> str:

        response = self.client.chat.completions.create(
            model=self.config.model_id,
            response_format={"type": "json_object"} if json_mode else {"type": "text"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            stream=True,
            extra_body={"enable_thinking": enable_thinking},
        )

        full_content = []
        has_printed_separator = False
        print(f"\n{'='*15} LLM 响应开始 {'='*15}\n")

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", "") or ""
            content = delta.content or ""

            if reasoning:
                self._print_stream(reasoning)
            elif content:
                if not has_printed_separator:
                    print(f"\n\n{'='*15} 最终回答 {'='*15}\n")
                    has_printed_separator = True
                self._print_stream(content)
                full_content.append(content)

        print(f"\n\n{'='*15} 响应结束 {'='*15}\n")
        return "".join(full_content)


class BaseAgent(ABC):
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    # 子类必须实现的统一接口
    @abstractmethod
    def process_task(self, message: McpMessage) -> McpMessage:
        pass


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
spec = ServerlessSpec(cloud="aws", region="us-east-1")

# INDEX_NAME 就是为这个 Pinecone 索引设定的 “专属名称”，相当于数据库中 “表名” 的角色 ——
# 通过这个名称，智能体（如 Researcher Agent、Context Librarian Agent）能精准定位到需要查询或写入数据的向量集合，确保数据流转的准确性。
INDEX_NAME = "genai-mas-mcp-ch3"
dimension_str = os.getenv("EMBEDDING_DIM")
if dimension_str is None:
    raise ValueError("EMBEDDING_DIM environment variable is not set.")
dimension = int(dimension_str)

if INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{INDEX_NAME}' not found, Creating new Serverless index...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=dimension,
        metric="cosine",  # 余弦相似度
        spec=spec,
    )
    while True:
        status = pc.describe_index(INDEX_NAME).status
        if status is not None and status["ready"]:
            break
        print("Waiting for index to be ready...")
        time.sleep(1)

    print("Index created successfully. It is new and empty")
else:
    print(
        f"Index '{INDEX_NAME}' already exists. Clearing namespaces for a fresh start.."
    )
    index = pc.Index(INDEX_NAME)
    namespaces_to_clear = [
        os.getenv("NAMESPACE_KNOWLEDGE"),
        os.getenv("NAMESPACE_CONTEXT"),
    ]
    for namespace in namespaces_to_clear:
        stats = index.describe_index_stats()
        if (
            namespace in stats.namespaces
            and stats.namespaces[namespace].vector_count > 0
        ):
            print(f"Clearing namespace '{namespace}'...")
            index.delete(delete_all=True, namespace=namespace)

            # 这里有个棘手的问题。清理函数可能是异步的，程序可能会进展得太快。
            # 这意味着，缓慢的删除操作可能在我们的上传开始后才完成，而那些新向量可能会被清除掉。
            # 这并非必然，但无疑是我们不愿承担的风险。因此，我们强制系统等待，只有当向量数量真正为零时才继续执行：
            while True:
                stats = index.describe_index_stats()
                if (
                    namespace not in stats.namespaces
                    or stats.namespaces[namespace].vector_count == 0
                ):
                    print(f"Namespace '{namespace}' cleared successfully.")
                    break
                print(f"Waiting for namespace '{namespace}' to clear...")
                time.sleep(5)  # Poll every 5 seconds
        else:
            print(
                f"Namespace '{namespace}' is already empty or does not exist. Skipping."
            )

index = pc.Index(INDEX_NAME)

context_blueprints = [
    {
        "id": "blueprint_suspense_narrative",
        "description": "A precise Semantic Blueprint designed to generate suspenseful and tense narratives, suitable for children's stories. Focuses on atmosphere, perceived threats, and emotional impact. Ideal for creative writing.",
        "blueprint": json.dumps(
            {
                "scene_goal": "Increase tension and create suspense.",
                "style_guide": "Use short, sharp sentences. Focus on sensory details (sounds, shadows). Maintain a slightly eerie but age-appropriate tone.",
                "participants": [
                    {
                        "role": "Agent",
                        "description": "The protagonist experiencing the events.",
                    },
                    {
                        "role": "Source_of_Threat",
                        "description": "The underlying danger or mystery.",
                    },
                ],
                "instruction": "Rewrite the provided facts into a narrative adhering strictly to the scene_goal and style_guide.",
            }
        ),
    },
    {
        "id": "blueprint_technical_explanation",
        "description": "A Semantic Blueprint designed for technical explanation or analysis. This blueprint focuses on clarity, objectivity, and structure. Ideal for breaking down complex processes, explaining mechanisms, or summarizing scientific findings.",
        "blueprint": json.dumps(
            {
                "scene_goal": "Explain the mechanism or findings clearly and concisely.",
                "style_guide": "Maintain an objective and formal tone. Use precise terminology. Prioritize factual accuracy and clarity over narrative flair.",
                "structure": [
                    "Definition",
                    "Function/Operation",
                    "Key Findings/Impact",
                ],
                "instruction": "Organize the provided facts into the defined structure, adhering to the style_guide.",
            }
        ),
    },
    {
        "id": "blueprint_casual_summary",
        "description": "A goal-oriented context for creating a casual, easy-to-read summary. Focuses on brevity and accessibility, explaining concepts simply.",
        "blueprint": json.dumps(
            {
                "scene_goal": "Summarize information quickly and casually.",
                "style_guide": "Use informal language. Keep it brief and engaging. Imagine explaining it to a friend.",
                "instruction": "Summarize the provided facts using the casual style guide.",
            }
        ),
    },
]
# @title 4.Data Preparation: The Knowledge Base (Factual RAG)
# -------------------------------------------------------------------------
# We use sample data related to space exploration.

knowledge_data_raw = """
Space exploration is the use of astronomy and space technology to explore outer space. The early era of space exploration was driven by a "Space Race" between the Soviet Union and the United States. The launch of the Soviet Union's Sputnik 1 in 1957, and the first Moon landing by the American Apollo 11 mission in 1969 are key landmarks.

The Apollo program was the United States human spaceflight program carried out by NASA which succeeded in landing the first humans on the Moon. Apollo 11 was the first mission to land on the Moon, commanded by Neil Armstrong and lunar module pilot Buzz Aldrin, with Michael Collins as command module pilot. Armstrong's first step onto the lunar surface occurred on July 20, 1969, and was broadcast on live TV worldwide. The landing required Armstrong to take manual control of the Lunar Module Eagle due to navigational challenges and low fuel.

Juno is a NASA space probe orbiting the planet Jupiter. It was launched on August 5, 2011, and entered a polar orbit of Jupiter on July 5, 2016. Juno's mission is to measure Jupiter's composition, gravitational field, magnetic field, and polar magnetosphere to understand how the planet formed. Juno is the second spacecraft to orbit Jupiter, after the Galileo orbiter. It is uniquely powered by large solar arrays instead of RTGs (Radioisotope Thermoelectric Generators), making it the farthest solar-powered mission.

A Mars rover is a remote-controlled motor vehicle designed to travel on the surface of Mars. NASA JPL managed several successful rovers including: Sojourner, Spirit, Opportunity, Curiosity, and Perseverance. The search for evidence of habitability and organic carbon on Mars is now a primary NASA objective. Perseverance also carried the Ingenuity helicopter.
"""

tokenizer = tiktoken.get_encoding("cl100k_base")


def chunk_text(text, chunk_size=400, overlap=50):
    # 将原始字符串文本转换为模型可识别的令牌（token）序列
    tokens = tokenizer.encode(text)
    chunks = []
    # 令牌序列：0—100—200—300—350—400—500—600—700—750—800—900—1000
    # 第1片段：[0──────────────400)  （覆盖0-399令牌）
    # 第2片段：        [350──────────────750)  （覆盖350-749令牌，与第1片段重叠50个）
    # 第3片段：                [700──────────────1000)  （覆盖700-999令牌，与第2片段重叠50个）
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        # 将切分后的令牌子序列chunk_tokens（整数列表），反向转换为人类可阅读的字符串文本，这是tokenizer.encode()的逆操作。
        chunk_text = tokenizer.decode(chunk_tokens)
        chunk_text = chunk_text.replace("\n", " ").strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embeddings_batch(texts, model="Qwen/Qwen3-Embedding-8B"):
    """Generates embeddings for a batch of texts using OpenAI, with retries."""
    if model is None:
        raise ValueError(
            "模型名称不能为空！请配置环境变量EMBEDDING_MODEL，或手动传入有效嵌入模型名称（如text-embedding-3-small）"
        )

    # OpenAI expects the input texts to have newlines replaced by spaces
    texts = [t.replace("\n", " ") for t in texts]
    config = AppConfig()
    llm = LLMService(config)
    dimensions_str = os.getenv("EMBEDDING_DIM")
    if dimensions_str is None:
        raise ValueError("EMBEDDING_DIM environment variable is not set.")

    dimensions = int(dimensions_str)
    response = llm.client.embeddings.create(
        input=texts, model=model, dimensions=dimensions
    )
    return [item.embedding for item in response.data]


vectors_context = []
for item in tqdm(context_blueprints):
    embedding = get_embeddings_batch([item["description"]])[0]
    vectors_context.append(
        {
            "id": item["id"],
            "values": embedding,
            "metadata": {
                "description": item["description"],
                # The blueprint itself (JSON string) is stored as metadata
                "blueprint_json": item["blueprint"],
            },
        }
    )

# Upsert data
if vectors_context:
    index.upsert(vectors=vectors_context, namespace=os.getenv("NAMESPACE_CONTEXT"))
    print(f"Successfully uploaded {len(vectors_context)} context vectors.")

# --- 6.2. Knowledge Base ---
print(
    f"\nProcessing and uploading Knowledge Base to namespace: {os.getenv("NAMESPACE_KNOWLEDGE")}"
)

# --- 6.2. Knowledge Base ---
print(
    f"\nProcessing and uploading Knowledge Base to namespace: {os.getenv("NAMESPACE_KNOWLEDGE")}"
)

# Chunk the knowledge data
knowledge_chunks = chunk_text(knowledge_data_raw)
print(f"Created {len(knowledge_chunks)} knowledge chunks.")

vectors_knowledge = []
batch_size = 100  # Process in batches

for i in tqdm(range(0, len(knowledge_chunks), batch_size)):
    batch_texts = knowledge_chunks[i : i + batch_size]
    batch_embeddings = get_embeddings_batch(batch_texts)

    batch_vectors = []
    for j, embedding in enumerate(batch_embeddings):
        chunk_id = f"knowledge_chunk_{i+j}"
        batch_vectors.append(
            {"id": chunk_id, "values": embedding, "metadata": {"text": batch_texts[j]}}
        )
    # Upsert the batch
    index.upsert(vectors=batch_vectors, namespace=os.getenv("NAMESPACE_KNOWLEDGE"))

print(f"Successfully uploaded {len(knowledge_chunks)} knowledge vectors.")


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding(text):
    """Generates embeddings for a single text query with retries."""
    text = text.replace("\n", " ")
    config = AppConfig()
    llm = LLMService(config)
    dimensions_str = os.getenv("EMBEDDING_DIM")
    if dimensions_str is None:
        raise ValueError("EMBEDDING_DIM environment variable is not set.")

    dimensions = int(dimensions_str)
    response = llm.client.embeddings.create(
        input=[text], model="Qwen/Qwen3-Embedding-8B", dimensions=dimensions
    )
    return response.data[0].embedding


def query_pinecone(query_text, namespace, top_k=1):
    """Embeds the query text and searches the specified Pinecone namespace."""
    try:
        query_embedding = get_embedding(query_text)
        response = index.query(
            vector=query_embedding,
            namespace=namespace,
            top_k=top_k,
            include_metadata=True,
        )
        return response["matches"]
    except Exception as e:
        print(f"Error querying Pinecone (Namespace: {namespace}): {e}")
        return []


# 从向量库NAMESPACE_CONTEXT中查找符合语义的蓝图手册
class agent_context_librarian(BaseAgent):
    def process_task(self, message: McpMessage) -> McpMessage:
        print("上下文管理员激活")
        if isinstance(message.content, str):
            data = json.loads(message.content)
        else:
            data = message.content
        requested_intent = data.get("intent_query", "")
        results = query_pinecone(
            requested_intent, os.getenv("NAMESPACE_CONTEXT"), top_k=1
        )
        if results:
            match = results[0]
            print(
                f"上下文管理员找到蓝图了 '{match['id']}' (得分：{match['score']: .2f})"
            )
            blueprint_json = match["metadata"]["blueprint_json"]
            content = {"blueprint": blueprint_json}
        else:
            print("上下文管理员没有找到蓝图，返回默认的")
            content = {
                "blueprint": json.dumps(
                    {"instruction": "Generate the content neutrally"}
                )
            }

        return McpMessage(sender="Librarian", content=content)


class agent_researcher(BaseAgent):
    def process_task(self, message: McpMessage) -> McpMessage:
        if isinstance(message.content, str):
            data = json.loads(message.content)
        else:
            data = message.content
        topic = data.get("topic_query")
        results = query_pinecone(topic, os.getenv("NAMESPACE_KNOWLEDGE"), top_k=3)

        if not results:
            print("[Researcher] No relevant information found")
            return McpMessage(sender="Researcher", content={"fact": "No data found"})

        source_texts = [match["metadata"]["text"] for match in results]

        system_prompt = """You are an expert research synthesis AI.Synthesize the provided source texts into a concise, bullet-pointed summary relevant to the user's topic. Focus strictly on the facts provided in the sources. Do not add outside information."""

        user_prompt = f"Topic: {topic}\n\nSources:\n" + "\n\n--\n\n".join(source_texts)

        findings = self.llm.chat_completion(system_prompt, user_prompt)

        return McpMessage(sender="Researcher", content={"facts": findings})


class agent_writer(BaseAgent):
    def process_task(self, message: McpMessage) -> McpMessage:
        if isinstance(message.content, str):
            data = json.loads(message.content)
        else:
            data = message.content
        facts = data.get("facts")
        blueprint_json_string = data.get("blueprint")

        # The Writer's System Prompt incorporates the dynamically retrieved blueprint
        system_prompt = f"""You are an expert content generation AI.
        Your task is to generate content based on the provided RESEARCH FINDINGS.
        Crucially, you MUST structure, style, and constrain your output according to the rules defined in the SEMANTIC BLUEPRINT provided below.

        --- SEMANTIC BLUEPRINT (JSON) ---
        {blueprint_json_string}
        --- END SEMANTIC BLUEPRINT ---

        Adhere strictly to the blueprint's instructions, style guides, and goals. The blueprint defines HOW you write; the research defines WHAT you write about.
        """

        user_prompt = f"""
        --- RESEARCH FINDINGS ---
        {facts}
        --- END RESEARCH FINDINGS ---

        Generate the content now.
        """
        final_output = self.llm.chat_completion(system_prompt, user_prompt)

        return McpMessage(sender="Writer", content={"output": final_output})


class Orchestrator(BaseAgent):
    def __init__(self, llm_service: LLMService, high_level_goal: str):
        super().__init__(llm_service)
        self.high_level_goal = high_level_goal

    """
    Manages the workflow of the Context-Aware MAS.
    Analyzes the goal, retrieves context and facts, and coordinates generation.
    """

    def process_task(self, message: McpMessage) -> McpMessage:
        print(f"=== [Orchestrator] Starting New Task ===")
        print(f"Goal: {self.high_level_goal}")

        # Step 0: Analyze Goal (Determine Intent and Topic)
        # We use the LLM to separate the desired style (intent) from the subject matter (topic).
        print("\n[Orchestrator] Analyzing Goal...")
        analysis_system_prompt = """You are an expert goal analyst. Analyze the user's high-level goal and extract two components:
            1. 'intent_query': A descriptive phrase summarizing the desired style, tone, or format, optimized for searching a context library (e.g., "suspenseful narrative blueprint", "objective technical explanation structure").
            2. 'topic_query': A concise phrase summarizing the factual subject matter required (e.g., "Juno mission objectives and power", "Apollo 11 landing details").

            Respond ONLY with a JSON object containing these two keys."""

        analysis_result = self.llm.chat_completion(
            analysis_system_prompt, self.high_level_goal, json_mode=True
        )

        try:
            analysis = json.loads(analysis_result)
            intent_query = analysis["intent_query"]
            topic_query = analysis["topic_query"]
        except (json.JSONDecodeError, KeyError):
            print(
                f"[Orchestrator] Error: Could not parse analysis JSON. Raw Analysis: {analysis_result}. Aborting."
            )
            return McpMessage(sender="Orchestrator", content={})

        print(f"Orchestrator: Intent Query: '{intent_query}'")
        print(f"Orchestrator: Topic Query: '{topic_query}'")

        mcp_to_librarian = McpMessage(
            sender="Orchestrator", content={"intent_query": intent_query}
        )
        config = AppConfig()
        llm = LLMService(config)

        agentContextLibrarian = agent_context_librarian(llm)
        mcp_from_librarian = agentContextLibrarian.process_task(mcp_to_librarian)

        # Ensure content is a dict before calling .get
        content = mcp_from_librarian.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except Exception:
                content = {}
        context_blueprint = content.get("blueprint")

        if not context_blueprint:
            return McpMessage(sender="Orchestrator", content={})

        mcp_to_researcher = McpMessage(
            sender="Orchestrator", content={"topic_query": topic_query}
        )
        agentResearcher = agent_researcher(llm)
        mcp_from_researcher = agentResearcher.process_task(mcp_to_researcher)
        content = mcp_from_researcher.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except Exception:
                content = {}
        research_findings = content.get("facts")
        if not research_findings:
            return McpMessage(sender="Orchestrator", content={})

        writer_task = {"blueprint": context_blueprint, "facts": research_findings}
        mcp_to_writer = McpMessage(sender="Orchestrator", content=writer_task)
        agentWriter = agent_writer(llm)
        mcp_from_writer = agentWriter.process_task(mcp_to_writer)
        content = mcp_from_writer.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except Exception:
                content = {}
        final_result = content.get("output")

        return McpMessage(sender="Orchestrator", content={"output": final_result})


def main():
    try:
        config = AppConfig()
        llm = LLMService(config)
        orchestrator = Orchestrator(
            llm,
            high_level_goal="基于真实事实，创作一个关于朱诺号木星探测任务的悬疑故事。中文说明",
        )

        input_msg = McpMessage(sender="User", content="Mediterranean Diet")
        print(f"任务启动: {input_msg.content}")

        final_msg = orchestrator.process_task(input_msg)

        print("\n" + "#" * 30)
        print("最终结果：")
        content = final_msg.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except Exception:
                content = {}
        print(content.get("output"))
        print("#" * 30)

    except Exception as e:
        logger.exception("程序发生致命错误")
        sys.exit(1)


if __name__ == "__main__":
    main()
