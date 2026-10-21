"""
轻量版知识点体系构建（T4 前置，供 T2 异质图使用）
完整版 T4（知识点依赖链 + DKT 逐题序列）后续再补

当前做的事：
  1) 从 quiz_item.parquet 提取所有 course_code × kp_id 对 → 生成知识点节点表
  2) 生成 (course, has_kt, kp) 边
  3) course_prereq 直接读清洗好的 course_prereq.parquet → 课程先修边
后续要补的：
  - kp_dependency 知识点内部依赖链
  - DKT 逐题答题序列
  - item_kp_map 完整题库→知识点映射
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INTERIM = Path("data/interim")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


def build_knowledge_point_nodes() -> pd.DataFrame:
    """从 quiz_item 去重提取 (course_code, kp_id) → 知识点节点
    补充 mock 里 quiz 可能没覆盖的课程的知识点（按 KP_PER_COURSE 标准补齐）
    """
    quiz = pd.read_parquet(INTERIM / "quiz_item.parquet")

    # 从有 kp_id 的小测题里提取知识点
    kp_from_quiz = quiz[quiz["kp_id"].notna()][["course_code", "kp_id"]].drop_duplicates()

    # 补齐所有课程的 10 个知识点（mock 约定 KP_PER_COURSE=10）
    courses = pd.read_parquet(INTERIM / "courses.parquet")["code"].unique()
    all_kps = []
    for code in courses:
        for i in range(1, 11):
            all_kps.append({"kp_id": f"kp_{code}_{i:02d}", "course_code": code})
    df_kps = pd.DataFrame(all_kps)

    # 标记哪些是 quiz 里实际出现过的
    df_kps = df_kps.merge(
        kp_from_quiz.assign(in_quiz=True), on=["kp_id", "course_code"], how="left",
    )
    df_kps["in_quiz"] = df_kps["in_quiz"].fillna(False)

    # 难度 proxy：先用 0.5 占位，后续可以从 quiz 正确率反推
    df_kps["difficulty_proxy"] = 0.5
    df_kps["suggested_week"] = df_kps.index % 16  # 简化占位
    df_kps["name"] = df_kps["kp_id"]

    return df_kps


def build_course_has_kt_edges(df_kps: pd.DataFrame) -> pd.DataFrame:
    """(course, has_kt, knowledge_point) 边"""
    edges = df_kps[["course_code", "kp_id"]].copy()
    edges.columns = ["src", "dst"]
    edges["src_type"] = "course"
    edges["dst_type"] = "knowledge_point"
    return edges


def build_kp_dependency_edges(df_kps: pd.DataFrame) -> pd.DataFrame:
    """知识点内部依赖链：同一课程内按编号顺序，小 → 大
    这是简化版，后续完整版会结合先修关系 + 人工字典
    """
    rows = []
    for course, sub in df_kps.groupby("course_code"):
        sub = sub.sort_values("kp_id").reset_index(drop=True)
        for i in range(len(sub) - 1):
            rows.append({
                "src": sub.at[i, "kp_id"],
                "dst": sub.at[i + 1, "kp_id"],
                "src_type": "knowledge_point",
                "dst_type": "knowledge_point",
                "source": "rule",
                "strength": 0.7,
            })
    return pd.DataFrame(rows)


def main():
    print("Building knowledge point nodes...")
    df_kps = build_knowledge_point_nodes()
    df_kps.to_parquet(OUT / "dim_knowledge_point.parquet", index=False)
    print(f"  knowledge_point nodes: {len(df_kps)}")

    print("Building (course, has_kt, kp) edges...")
    has_kt = build_course_has_kt_edges(df_kps)
    has_kt.to_parquet(OUT / "edge_course_has_kt.parquet", index=False)
    print(f"  has_kt edges: {len(has_kt)}")

    print("Building kp dependency edges (rule-based)...")
    kp_dep = build_kp_dependency_edges(df_kps)
    kp_dep.to_parquet(OUT / "edge_kp_prereq.parquet", index=False)
    print(f"  kp_prereq edges: {len(kp_dep)}")

    print("Done. 后续还需补：item_kp_map + DKT sequences + 人工知识点字典")


if __name__ == "__main__":
    main()
