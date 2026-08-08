#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
push_to_dify_kb.py — 将管道腐蚀研究资料推送到 Dify Cloud 知识库

【前置条件】
  本脚本使用的是 Dify 的「知识 API 密钥」(Knowledge API Key, 以 dataset- 开头),
  不是应用的 app- 密钥! 应用密钥只能调用问答接口, 不能管理知识库。
  获取方式: Dify 控制台 -> 左侧「Knowledge(知识库)」-> 右上角「Service API」-> 创建密钥。

【用法】
  # 1) 设置密钥(也可写入 .env 或直接在下方 DEFAULT_API_KEY 填)
  export DIFY_KB_API_KEY="dataset-xxxxxx"

  # 2) 运行(默认会列出知识库, 不存在则新建, 然后上传两份资料)
  python scripts/push_to_dify_kb.py

  # 仅列出知识库, 不上传:
  python scripts/push_to_dify_kb.py --list-only

  # 指定已有知识库 ID 直接上传(跳过自动匹配):
  python scripts/push_to_dify_kb.py --dataset-id <KB_ID>

【说明】
  - 默认索引方式 high_quality(向量, 需知识库已配置嵌入模型)。
    若报嵌入模型相关错误, 加参数 --economy 改用 economy(关键词, 免嵌入模型)。
  - 文档上传为异步处理, 脚本仅提交并返回批次号, Dify 后台会索引。
  - 已链接到应用的同一知识库, 新增文档后无需重新发布应用即可被问答检索到。
"""

import os
import sys
import argparse
import requests

# ============ 配置 ============
DEFAULT_BASE_URL = "https://api.dify.ai/v1"
# 也可把密钥直接填这里(不推荐提交到 git, 建议用环境变量)
DEFAULT_API_KEY = os.environ.get("DIFY_KB_API_KEY", "")

# 知识库名称(不存在时自动创建)
KB_NAME = "管道腐蚀标准与文献知识库"
KB_DESCRIPTION = "管道腐蚀国际标准(NACE/API/ASME)、中国标准(GB/SY/T)、研究资料与公开数据源(PHMSA/PRCI/EGIG 等)、细分条款与失效案例。"

# 要上传的文件(相对项目根目录)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILES_TO_UPLOAD = [
    os.path.join(PROJECT_ROOT, "data", "standards", "research_references.md"),
    os.path.join(PROJECT_ROOT, "data", "standards", "china_standards_clauses.md"),
]


def _headers(api_key):
    return {"Authorization": f"Bearer {api_key}"}


def list_datasets(base_url, api_key):
    """列出所有知识库, 返回 (list, total)"""
    out = []
    page = 1
    while True:
        r = requests.get(
            f"{base_url}/datasets",
            headers=_headers(api_key),
            params={"page": page, "limit": 100},
            timeout=(10, 30),
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        out.extend(data)
        total_pages = r.json().get("total_pages", 1)
        if page >= total_pages or not data:
            break
        page += 1
    return out


def create_dataset(base_url, api_key, name, description):
    r = requests.post(
        f"{base_url}/datasets",
        headers={**_headers(api_key), "Content-Type": "application/json"},
        json={
            "name": name,
            "description": description,
            "indexing_technique": "high_quality",
            "permission": "only_me",
        },
        timeout=(10, 30),
    )
    r.raise_for_status()
    return r.json()


def list_documents(base_url, api_key, dataset_id):
    """列出某知识库的全部文档, 返回 list"""
    out = []
    page = 1
    while True:
        r = requests.get(
            f"{base_url}/datasets/{dataset_id}/documents",
            headers=_headers(api_key),
            params={"page": page, "limit": 100},
            timeout=(10, 30),
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        out.extend(data)
        total_pages = r.json().get("total_pages", 1)
        if page >= total_pages or not data:
            break
        page += 1
    return out


def delete_document(base_url, api_key, dataset_id, doc_id):
    r = requests.delete(
        f"{base_url}/datasets/{dataset_id}/documents/{doc_id}",
        headers=_headers(api_key),
        timeout=(10, 30),
    )
    r.raise_for_status()
    # Dify DELETE 常返回 204 空 body, 不能 json() 解析
    try:
        return r.json()
    except ValueError:
        return {"status": "deleted", "id": doc_id}


def upload_document(base_url, api_key, dataset_id, file_path, indexing_technique):
    """按文件创建文档(自动匹配 v2/v1 端点)"""
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "text/markdown")}
        data = {
            "data": '{"indexing_technique": "%s", "process_rule": {"mode": "automatic"}}'
            % indexing_technique
        }
        last_err = None
        for endpoint in (
            f"{base_url}/datasets/{dataset_id}/documents/create-by-file",
            f"{base_url}/datasets/{dataset_id}/document/create-by-file",
        ):
            try:
                r = requests.post(
                    endpoint,
                    headers=_headers(api_key),
                    files=files,
                    data=data,
                    timeout=(10, 120),
                )
                if r.status_code == 404:
                    last_err = "404"
                    continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.HTTPError as e:
                last_err = str(e)
                # 尝试下一个端点
                continue
            except Exception as e:
                last_err = str(e)
                continue
        raise RuntimeError(f"上传失败 (endpoints tried, last error: {last_err})")


def main():
    parser = argparse.ArgumentParser(description="推送研究资料到 Dify 知识库")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="知识 API 密钥 (dataset-)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Dify API base URL")
    parser.add_argument("--dataset-id", default=None, help="指定知识库 ID (跳过自动匹配)")
    parser.add_argument("--list-only", action="store_true", help="仅列出知识库")
    parser.add_argument("--economy", action="store_true", help="使用 economy 索引(免嵌入模型)")
    parser.add_argument("--refresh", action="store_true",
                        help="上传前先删除知识库里同名旧文档(避免重复堆积)")
    args = parser.parse_args()

    if not args.api_key:
        print("❌ 未提供知识 API 密钥。")
        print("   获取: Dify 控制台 -> Knowledge -> 右上角 Service API -> 创建密钥(dataset- 开头)")
        print("   设置: export DIFY_KB_API_KEY=\"dataset-xxxx\"  或编辑本脚本 DEFAULT_API_KEY")
        sys.exit(1)

    indexing = "economy" if args.economy else "high_quality"

    try:
        datasets = list_datasets(args.base_url, args.api_key)
    except Exception as e:
        print(f"❌ 列出知识库失败: {e}")
        print("   请确认: 1) 密钥是 dataset- 开头的「知识 API 密钥」(不是 app-); 2) 网络可访问 api.dify.ai")
        sys.exit(1)

    print(f"📚 已有知识库 ({len(datasets)} 个):")
    for d in datasets:
        print(f"   - {d.get('name')}  (id={d.get('id')})")

    if args.list_only:
        return

    # 选择知识库
    target = None
    if args.dataset_id:
        target = next((d for d in datasets if d.get("id") == args.dataset_id), None)
        if not target:
            print(f"⚠️ 未找到指定 ID 的知识库, 将新建: {KB_NAME}")
    if target is None:
        target = next((d for d in datasets if d.get("name") == KB_NAME), None)
    if target is None:
        print(f"➕ 未找到「{KB_NAME}」, 新建知识库...")
        try:
            target = create_dataset(args.base_url, args.api_key, KB_NAME, KB_DESCRIPTION)
            print(f"   ✅ 已创建: {target.get('id')}")
        except Exception as e:
            print(f"❌ 创建知识库失败: {e}")
            sys.exit(1)
    else:
        print(f"✅ 使用知识库: {target.get('name')} ({target.get('id')})")

    dataset_id = target.get("id")

    # 刷新模式: 删除同名旧文档
    if args.refresh:
        try:
            existing = list_documents(args.base_url, args.api_key, dataset_id)
            base_names = {os.path.basename(fp) for fp in FILES_TO_UPLOAD}
            to_delete = [d for d in existing if d.get("name") in base_names]
            if to_delete:
                print(f"\n🧹 刷新模式: 删除 {len(to_delete)} 个同名旧文档...")
                for d in to_delete:
                    try:
                        delete_document(args.base_url, args.api_key, dataset_id, d.get("id"))
                        print(f"   🗑️ 已删除: {d.get('name')} ({d.get('id')})")
                    except Exception as e:
                        print(f"   ⚠️ 删除失败 {d.get('name')}: {e}")
            else:
                print("\n🧹 刷新模式: 无同名旧文档, 直接上传。")
        except Exception as e:
            print(f"⚠️ 列出/删除旧文档失败(继续上传): {e}")

    # 上传文件
    for fp in FILES_TO_UPLOAD:
        if not os.path.exists(fp):
            print(f"⚠️ 跳过(文件不存在): {fp}")
            continue
        print(f"\n📤 上传: {os.path.basename(fp)}  (索引: {indexing})")
        try:
            res = upload_document(args.base_url, args.api_key, dataset_id, fp, indexing)
            doc = res.get("document", {})
            print(f"   ✅ 已提交 | 文档ID={doc.get('id')} | 批次={res.get('batch')} | 状态={doc.get('status') or doc.get('indexing_status')}")
        except Exception as e:
            print(f"   ❌ 上传失败: {e}")
            if "embedding" in str(e).lower() or "high_quality" in str(e):
                print("   💡 提示: 若报嵌入模型错误, 请加 --economy 参数重跑。")
            print("   继续下一个文件...")

    print("\n✅ 完成。Dify 后台将异步索引, 几分钟后应用问答即可覆盖新资料。")


if __name__ == "__main__":
    main()
