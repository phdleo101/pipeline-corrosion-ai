"""
rag_engine.py
管道腐蚀标准知识库 RAG 引擎
- 支持本地向量检索 (ChromaDB) 和 Dify Cloud API 两种模式
- 支持 blocking（同步）和 streaming（流式）两种查询模式
- 文档加载、分段、向量化、检索、问答
"""

import os
import json
import time
import requests
from collections import OrderedDict


class _LRUCache:
    """简单的 LRU 缓存，用于缓存常见问题回答"""

    def __init__(self, capacity=20):
        self._cache = OrderedDict()
        self._capacity = capacity

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)


def _load_dify_config():
    """
    加载 Dify 配置（支持多种来源）

    优先级:
    1. Streamlit Secrets（部署时推荐）
    2. 环境变量（本地开发时使用）
    3. app_config.yaml 文件
    """
    config = {"api_url": "", "api_key": ""}

    # 1. 尝试从 Streamlit Secrets 读取（生产环境）
    try:
        import streamlit as st
        if "dify" in st.secrets:
            config["api_url"] = st.secrets["dify"].get("api_url", "")
            config["api_key"] = st.secrets["dify"].get("api_key", "")
            if config["api_url"] and config["api_key"]:
                return config
    except Exception:
        pass

    # 2. 尝试从环境变量读取
    config["api_url"] = os.environ.get("DIFY_API_URL", "")
    config["api_key"] = os.environ.get("DIFY_API_KEY", "")

    if config["api_url"] and config["api_key"]:
        return config

    # 3. 尝试从 YAML 配置文件读取（本地开发）
    try:
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "app_config.yaml"
        )
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}
            dify_cfg = yaml_config.get("dify", {})
            config["api_url"] = dify_cfg.get("api_url", "")
            config["api_key"] = dify_cfg.get("api_key", "")
    except Exception:
        pass

    return config


class CorrosionRAG:
    """管道腐蚀标准知识库问答引擎"""

    def __init__(self, config_path=None):
        """
        初始化 RAG 引擎

        模式优先级:
        1. Dify API 模式 (如果配置了 DIFY_API_URL 和 DIFY_API_KEY)
        2. 本地向量检索模式 (如果有标准文档)
        3. 降级模式 (返回提示信息)
        """
        self.mode = "fallback"
        self.vector_store = None
        self.llm = None
        self._cache = _LRUCache(capacity=20)  # 缓存常见问题回答

        # 加载配置（兼容多种来源）
        self.config = self._load_legacy_config(config_path)
        dify_cfg = _load_dify_config()
        dify_url = dify_cfg.get("api_url", "")
        dify_key = dify_cfg.get("api_key", "")

        if dify_url and dify_key:
            self.mode = "dify"
            self.dify_url = dify_url
            self.dify_key = dify_key
            print(f"[RAG] 使用 Dify Cloud API 模式 (URL: {dify_url})")
            return

        # 尝试初始化本地模式
        try:
            self._init_local_mode()
        except Exception as e:
            print(f"[RAG] 本地模式初始化失败: {e}")
            print("[RAG] 使用降级模式（需要配置 Dify API 或安装依赖）")

    def _load_legacy_config(self, config_path):
        """加载本地配置文件（用于本地 LLM key）"""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "app_config.yaml"
            )

        config = {}
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
            except ImportError:
                pass
        return config

    def _init_local_mode(self):
        """初始化本地向量检索模式"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import (
            HuggingFaceEmbeddings,
        )

        standards_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "standards"
        )
        standards_dir = os.path.abspath(standards_dir)

        documents = []
        text_files = []
        if os.path.exists(standards_dir):
            for f in os.listdir(standards_dir):
                if f.endswith((".txt", ".md")):
                    text_files.append(os.path.join(standards_dir, f))

        if not text_files:
            # 使用内置的腐蚀知识库
            documents = self._get_builtin_knowledge()
        else:
            from langchain_community.document_loaders import TextLoader
            for tf in text_files:
                loader = TextLoader(tf, encoding="utf-8")
                documents.extend(loader.load())

        if not documents:
            raise RuntimeError("没有找到标准文档")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", " "],
        )
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        db_path = os.path.join(os.path.dirname(__file__), "..", "models", "chroma_db")
        self.vector_store = Chroma.from_documents(
            texts, embeddings, persist_directory=db_path
        )

        # 初始化 LLM
        self._init_llm()

        self.mode = "local"
        print(f"[RAG] 使用本地向量检索模式，加载了 {len(texts)} 个文档段")

    def _init_llm(self):
        """初始化 LLM"""
        from langchain_community.llms import Tongyi
        from langchain.chains import RetrievalQA

        api_key = os.environ.get(
            "DASHSCOPE_API_KEY",
            self.config.get("dashscope_api_key", ""),
        )

        if api_key:
            self.llm = Tongyi(dashscope_api_key=api_key)
        else:
            # 尝试 OpenAI
            openai_key = os.environ.get("OPENAI_API_KEY", "")
            if openai_key:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    def _get_builtin_knowledge(self):
        """内置腐蚀知识库（当没有标准文档时使用）"""
        from langchain.schema import Document

        knowledge_base = [
            Document(
                page_content="""NACE MR0175/ISO 15156 材料选用标准：
在含硫化氢(H2S)环境中使用的油气管道材料选择标准。
当酸性气体环境中 H2S 分压超过 0.3 kPa (0.05 psi) 时，必须考虑抗硫化物应力开裂(SSC)材料。
碳钢在含 H2S 环境中需控制硬度不超过 22 HRC。
马氏体不锈钢在 H2S 环境中易发生 SSC，应避免使用或经过特殊热处理。
双相不锈钢在含氯离子和 H2S 环境中需控制使用温度和应力水平。""",
                metadata={"source": "NACE MR0175"},
            ),
            Document(
                page_content="""API 571 损伤机理：
管道常见损伤机理包括：腐蚀减薄（均匀腐蚀、局部腐蚀）、环境开裂（SCC、HIC、SSC）、
机械损伤（凹陷、沟槽）、蠕变变形、疲劳裂纹等。
CO2腐蚀（甜腐蚀）：当介质中含有 CO2 时，形成碳酸导致管道内壁腐蚀。
腐蚀速率随 CO2 分压增加而增大，温度在 60-100°C 范围内腐蚀最严重。
H2S腐蚀（酸腐蚀）：H2S 导致硫化物应力开裂(SSC)和氢致开裂(HIC)。
微生物诱导腐蚀(MIC)：硫酸盐还原菌(SRB)等微生物活动导致局部腐蚀。""",
                metadata={"source": "API 571"},
            ),
            Document(
                page_content="""ASME B31.8S 输气管道完整性管理：
管道完整性管理包括：基线评估、周期性再评估、持续监测、维修和风险降低措施。
风险评估应考虑：失效概率（腐蚀速率、缺陷尺寸、管材强度）和失效后果（人口密度、环境敏感性）。
高后果区(HCA)管道应优先评估，评估周期不超过 7 年。
完整性评估方法：内检测(ILI)、压力试验、直接评估(ECDA/ICDA)。""",
                metadata={"source": "ASME B31.8S"},
            ),
            Document(
                page_content="""ASME B31G 腐蚀缺陷评估：
用于评估管道金属损失缺陷的剩余强度。
B31G Level 1：简化评估，基于缺陷长度、深度和管道屈服强度计算剩余强度。
当缺陷深度超过壁厚的 80% 时，需立即维修或更换。
RSTRENG 方法是 B31G 的改进版，适用于复杂形状缺陷，提供更精确的评估结果。
安全系数：评估结果应考虑安全系数 1.25-2.0，取决于评估等级和后果严重性。""",
                metadata={"source": "ASME B31G"},
            ),
            Document(
                page_content="""阴极保护(CP)标准 NACE SP0162：
阴极保护是防止管道外腐蚀的主要手段，通过施加阴极电流使管道电位负移到保护电位。
最小保护电位准则：-850 mV (相对于 Cu/CuSO4 参比电极)。
最负电位限制：-1200 mV（避免过保护导致涂层剥离和氢脆）。
密间距测量(CIS)用于评估管道沿线的 CP 效果，间隔通常为 1-3 米。
DCVG（直流电压梯度）用于定位涂层缺陷，精度可达缺陷大小的 10%。""",
                metadata={"source": "NACE SP0162"},
            ),
            Document(
                page_content="""管道内检测(ILI)技术：
MFL（漏磁内检测）：检测管壁金属损失（腐蚀、机械损伤），可检测深度大于 10% 壁厚的缺陷。
UT（超声波内检测）：精确测量壁厚，适用于裂纹检测，需要液体耦合剂。
EMAT（电磁声换能器）：检测应力腐蚀开裂(SCC)和长条状缺陷，不需要耦合剂。
IMU（惯性测量单元）：检测管道几何变形和位移。
ILI 检测周期：高风险管道 5-7 年，中风险 7-10 年，低风险 10-14 年。""",
                metadata={"source": "API 1163"},
            ),
            Document(
                page_content="""腐蚀速率分类标准：
低风险：< 0.1 mm/a，腐蚀速率在可接受范围内，常规监测即可。
中风险：0.1 - 0.5 mm/a，需加强监测频率，考虑增加缓蚀剂。
高风险：0.5 - 1.0 mm/a，需评估剩余强度，制定维修计划。
严重风险：> 1.0 mm/a，需立即降压运行或维修，评估管道完整性。
缓蚀剂类型：吸附型（有机胺、咪唑啉）、沉淀型（磷酸盐、硅酸盐）、氧化型（铬酸盐）。
缓蚀剂注入效率评估：通过腐蚀挂片或在线探针监测注入前后的腐蚀速率变化。""",
                metadata={"source": "NACE SP0775"},
            ),
            Document(
                page_content="""管道腐蚀防护涂层系统：
三层PE（聚乙烯）涂层：环氧粉末底漆 + 粘结剂 + 聚乙烯外层，适用于大多数埋地管道。
FBE（熔结环氧粉末）：单层涂层，耐温性好，适用于高温管道。
液体环氧涂层：现场补口和修复使用。
涂层缺陷检测：使用 DCVG 或 Pearson 方法定位涂层破损点。
涂层老化评估：通过电化学阻抗谱(EIS)评估涂层防护性能衰减程度。""",
                metadata={"source": "NACE SP0185"},
            ),
        ]
        return knowledge_base

    def query(self, question):
        """
        查询知识库（同步模式，返回完整字符串）

        参数:
            question: 用户的自然语言问题

        返回:
            str: 答案文本
        """
        # 检查缓存
        cached = self._cache.get(question)
        if cached:
            print(f"[RAG] 命中缓存: {question[:30]}...")
            return cached

        if self.mode == "dify":
            result = self._query_dify(question)
        elif self.mode == "local":
            result = self._query_local(question)
        else:
            result = self._query_fallback(question)

        # 缓存结果（fallback 模式不缓存）
        if self.mode != "fallback" and result:
            self._cache.set(question, result)

        return result

    def query_stream(self, question):
        """
        流式查询知识库（返回生成器，逐字输出）

        适用于 Dify 模式，使用 SSE 流式传输。
        对于 local/fallback 模式，退化为同步查询后逐字模拟输出。

        参数:
            question: 用户的自然语言问题

        返回:
            generator: 逐段 yield 答案文本
        """
        # 检查缓存（命中缓存时直接逐字返回）
        cached = self._cache.get(question)
        if cached:
            print(f"[RAG] 流式命中缓存: {question[:30]}...")
            # 模拟逐字输出
            chunk_size = 10
            for i in range(0, len(cached), chunk_size):
                yield cached[i : i + chunk_size]
            return

        if self.mode == "dify":
            # 流式调用 Dify API
            full_answer = ""
            try:
                for chunk in self._query_dify_stream(question):
                    full_answer += chunk
                    yield chunk
                # 缓存完整回答
                if full_answer:
                    self._cache.set(question, full_answer)
            except Exception as e:
                error_msg = f"\n\n⚠️ 流式查询出错: {e}"
                yield error_msg
        elif self.mode == "local":
            result = self._query_local(question)
            self._cache.set(question, result)
            # 逐字模拟
            chunk_size = 10
            for i in range(0, len(result), chunk_size):
                yield result[i : i + chunk_size]
        else:
            result = self._query_fallback(question)
            yield result

    def _query_dify(self, question):
        """通过 Dify Cloud API 查询（带重试机制 + 长超时）"""
        import time

        headers = {
            "Authorization": f"Bearer {self.dify_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": {},
            "query": question,
            "response_mode": "blocking",
            "user": "corrosion-ai-user",
        }

        # 重试配置：3 次重试，指数退避
        max_retries = 3
        timeout_config = (10, 90)  # (连接超时, 读取超时)

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[RAG] Dify API 调用第 {attempt}/{max_retries} 次...")
                response = requests.post(
                    f"{self.dify_url}/chat-messages",
                    headers=headers,
                    json=payload,
                    timeout=timeout_config,
                )
                response.raise_for_status()
                result = response.json()
                answer = result.get("answer", "未能获取回答")
                if answer and answer != "未能获取回答":
                    print(f"[RAG] Dify API 调用成功（{attempt} 次尝试）")
                    return answer
                else:
                    last_error = "Dify 返回了空回答"
                    print(f"[RAG] {last_error}")
            except requests.exceptions.Timeout as e:
                last_error = f"网络超时（{timeout_config[1]}秒内无响应）"
                print(f"[RAG] {last_error}: {e}")
            except requests.exceptions.ConnectionError as e:
                last_error = f"网络连接错误"
                print(f"[RAG] {last_error}: {e}")
            except Exception as e:
                last_error = str(e)
                print(f"[RAG] 调用异常: {e}")

            # 如果不是最后一次，等待后重试
            if attempt < max_retries:
                wait_time = 2 ** attempt  # 2s, 4s, 8s
                print(f"[RAG] 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)

        # 3 次都失败
        error_msg = f"""Dify API 查询失败（已重试 {max_retries} 次）: {last_error}

**可能原因**:
1. Streamlit Cloud 出口网络访问 api.dify.ai 不稳定
2. Dify Cloud 服务端响应较慢
3. 网络临时波动

**建议**:
- 稍后重试（云端网络可能临时拥堵）
- 或在 Streamlit Cloud 控制台点击「Reboot app」重启应用
- 复杂查询（需检索多个段落）可能需要更长响应时间"""
        return error_msg

    def _query_dify_stream(self, question):
        """通过 Dify Cloud API 流式查询（SSE）"""
        headers = {
            "Authorization": f"Bearer {self.dify_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": {},
            "query": question,
            "response_mode": "streaming",
            "user": "corrosion-ai-user",
        }

        max_retries = 2
        timeout_config = (10, 120)  # 流式模式超时更宽松

        for attempt in range(1, max_retries + 1):
            try:
                print(f"[RAG] Dify Streaming API 调用第 {attempt}/{max_retries} 次...")
                response = requests.post(
                    f"{self.dify_url}/chat-messages",
                    headers=headers,
                    json=payload,
                    timeout=timeout_config,
                    stream=True,  # 启用流式接收
                )
                response.raise_for_status()

                # 解析 SSE 事件流
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]  # 去掉 "data: " 前缀
                        try:
                            data = json.loads(data_str)
                            event_type = data.get("event", "")

                            if event_type == "message":
                                answer_chunk = data.get("answer", "")
                                if answer_chunk:
                                    yield answer_chunk

                            elif event_type == "message_end":
                                print("[RAG] Dify Streaming 完成")
                                return

                            elif event_type == "error":
                                error_msg = data.get("message", "未知错误")
                                yield f"\n\n⚠️ Dify 返回错误: {error_msg}"
                                return

                        except json.JSONDecodeError:
                            continue

                # 如果循环正常结束（没有 message_end 事件）
                print("[RAG] Dify Streaming 流结束")
                return

            except requests.exceptions.Timeout:
                print(f"[RAG] 流式查询超时（尝试 {attempt}/{max_retries}）")
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    yield "\n\n⚠️ 网络超时，请稍后重试或点击清空对话后再次提问。"
                    return
            except requests.exceptions.ConnectionError:
                print(f"[RAG] 流式连接错误（尝试 {attempt}/{max_retries}）")
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    yield "\n\n⚠️ 网络连接失败，请稍后重试。"
                    return
            except Exception as e:
                print(f"[RAG] 流式查询异常: {e}")
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    yield f"\n\n⚠️ 查询出错: {e}"
                    return
        """通过本地向量检索查询"""
        try:
            if self.llm:
                from langchain.chains import RetrievalQA

                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                )
                result = qa_chain({"query": question})
                answer = result["result"]
                sources = result.get("source_documents", [])
                if sources:
                    source_names = set()
                    for doc in sources:
                        src = doc.metadata.get("source", "未知")
                        source_names.add(src)
                    answer += f"\n\n参考来源: {', '.join(source_names)}"
                return answer
            else:
                # 无 LLM 时返回检索到的文档片段
                docs = self.vector_store.similarity_search(question, k=3)
                if docs:
                    answer = "找到以下相关标准条款：\n\n"
                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get("source", "未知")
                        answer += f"[{i}] 来源: {source}\n{doc.page_content}\n\n"
                    answer += "\n（提示: 配置 LLM API Key 可获得更精准的问答能力）"
                    return answer
                return "未找到相关标准条款。"
        except Exception as e:
            return f"本地检索出错: {e}"

    def _query_fallback(self, question):
        """降级模式：返回配置指引"""
        return (
            "知识库引擎尚未配置。\n\n"
            "要启用智能问答功能，请选择以下任一方式：\n\n"
            "方式一：配置 Dify Cloud API（推荐）\n"
            "1. 注册 dify.ai 账号\n"
            "2. 创建知识库应用，上传标准文档\n"
            "3. 获取 API URL 和 API Key\n"
            "4. 设置环境变量：\n"
            "   export DIFY_API_URL='https://api.dify.ai/v1'\n"
            "   export DIFY_API_KEY='your-key'\n\n"
            "方式二：本地向量检索\n"
            "1. pip install langchain chromadb sentence-transformers\n"
            "2. 在 data/standards/ 目录放入标准文档（.txt 格式）\n"
            "3. 设置 LLM API Key（通义千问/OpenAI）\n\n"
            f"您的问题: {question}\n"
            "配置完成后即可获得回答。"
        )


if __name__ == "__main__":
    rag = CorrosionRAG()
    print(f"当前模式: {rag.mode}")

    test_questions = [
        "在什么条件下需要使用 NACE MR0175 规定的抗硫材料？",
        "CO2腐蚀的机理和影响因素是什么？",
        "管道腐蚀速率的风险等级如何划分？",
    ]

    for q in test_questions:
        print(f"\n问题: {q}")
        print(f"回答: {rag.query(q)}")
        print("-" * 60)
