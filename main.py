import os
import re
import csv
import json
import numpy as np
import torch
import string
import logging
import uuid
import threading
import math
import difflib
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_client import call_llm
from freebase_func import *
from chains_oneshot import *
from prompt_list import *


torch.manual_seed(42)
np.random.seed(42)

MODEL_SET = None

_global_model = None
_global_model_lock = threading.Lock()
_global_embeddings = None
_global_embeddings_lock = threading.Lock()
_print_lock = threading.Lock()
_schema_types_cache = None
_schema_types_lock = threading.Lock()
_type_embedding_cache = {}
_type_embedding_lock = threading.Lock()


def thread_safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def get_sentence_model():
    global _global_model
    if _global_model is None:
        with _global_model_lock:
            if _global_model is None:
                _global_model = SentenceTransformer(
                    "sentence-transformers/bge-large-en-v1.5",
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
    return _global_model


def get_global_embeddings():
    global _global_embeddings
    if _global_embeddings is None:
        with _global_embeddings_lock:
            if _global_embeddings is None:
                _global_embeddings = load_descriptions()
    return _global_embeddings


def load_schema_types(schema_path="freebase_schema.csv"):
    global _schema_types_cache
    if _schema_types_cache is not None:
        return _schema_types_cache
    with _schema_types_lock:
        if _schema_types_cache is not None:
            return _schema_types_cache
        domain_to_types = defaultdict(set)
        type_to_domain = {}
        all_types = set()
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        dom, pred, rng = row[0].strip(), row[1].strip(), row[2].strip()
                        if dom and '.' in dom:
                            all_types.add(dom)
                            prefix = dom.split('.')[0]
                            domain_to_types[prefix].add(dom)
                            type_to_domain[dom] = prefix
                        if rng and '.' in rng:
                            all_types.add(rng)
                            prefix = rng.split('.')[0]
                            domain_to_types[prefix].add(rng)
                            type_to_domain[rng] = prefix
        except:
            pass
        _schema_types_cache = {
            "domain_to_types": dict(domain_to_types),
            "type_to_domain": type_to_domain,
            "all_types": all_types
        }
        return _schema_types_cache


def normalize(s: str) -> str:
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = ' '.join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


def get_type_embedding_cached(type_str: str, model):
    if type_str in _type_embedding_cache:
        return _type_embedding_cache[type_str]
    with _type_embedding_lock:
        if type_str in _type_embedding_cache:
            return _type_embedding_cache[type_str]
        clean_text = type_str.replace('.', ' ').replace('_', ' ').strip()
        emb = model.encode([clean_text], normalize_embeddings=True, convert_to_numpy=True)[0]
        _type_embedding_cache[type_str] = emb
        return emb


def compute_type_similarity(type1: str, type2: str) -> float:
    if type1 == type2:
        return 1.0
    tokens1 = set(type1.replace('.', ' ').replace('_', ' ').lower().split())
    tokens2 = set(type2.replace('.', ' ').replace('_', ' ').lower().split())
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    jaccard = intersection / union if union > 0 else 0.0
    model = get_sentence_model()
    vec1 = get_type_embedding_cached(type1, model)
    vec2 = get_type_embedding_cached(type2, model)
    sem_score = float(np.dot(vec1, vec2))
    sem_score = max(0.0, sem_score)
    dom1 = type1.split('.')[0] if '.' in type1 else type1
    dom2 = type2.split('.')[0] if '.' in type2 else type2
    if dom1 == dom2:
        return jaccard + sem_score
    else:
        return sem_score


def is_type_compatible(end_type: str, answer_types: List[str], last_relation: str = None, trie=None) -> Tuple[
    bool, float]:
    final_type = end_type
    if (not final_type or final_type == "unknown") and last_relation and trie:
        meta = trie.get_relation_meta(last_relation)
        if meta["rng"] != "unknown":
            final_type = meta["rng"]
    if not answer_types:
        return True, 0.5
    max_sim = 0.0
    for at in answer_types:
        sim = compute_type_similarity(final_type, at)
        if sim > max_sim:
            max_sim = sim
    is_compatible = max_sim
    return is_compatible, max_sim


def normalize_entity_name(name):
    name = name.lower()
    name = re.sub(r'\W+', '_', name)
    return name.strip('_')


def safe_str(x):
    if isinstance(x, list):
        return " ".join(x)
    return str(x)


class SchemaStateNode:
    def __init__(self, relation=None, domain=None, range_val=None):
        self.relation = relation
        self.domain = domain
        self.range = range_val
        self.children = {}
        self.is_leaf = False

    def to_dict(self):
        return {
            "rel": self.relation,
            "schema": {"dom": self.domain, "rng": self.range},
            "children": {k: v.to_dict() for k, v in self.children.items()}
        }


class TypeAnchoredTrie:
    def __init__(self, schema_path="freebase_schema.csv"):
        self.pred_meta = {}
        self.roots = defaultdict(lambda: SchemaStateNode(relation="ROOT_TYPE"))
        self._load_schema(schema_path)

    def _load_schema(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        self.pred_meta[row[1].strip()] = {"dom": row[0].strip(), "rng": row[2].strip()}
        except Exception as e:
            print(f"Schema Load Error: {e}")

    def get_relation_meta(self, relation):
        return self.pred_meta.get(relation, {"dom": "unknown", "rng": "unknown"})

    def build_from_chains(self, chain_path):
        with open(chain_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                chain = [r.strip() for r in row]
                self._insert_chain_smart(chain)

    def _insert_chain_smart(self, chain):
        r1 = chain[0]
        meta = self.get_relation_meta(r1)
        valid_anchors = set()
        if meta["dom"] != "unknown":
            valid_anchors.add(meta["dom"])
        if meta["rng"] != "unknown":
            valid_anchors.add(meta["rng"])
        for anchor_type in valid_anchors:
            self._add_path_to_root(self.roots[anchor_type], chain)

    def _add_path_to_root(self, root_node, chain):
        current = root_node
        for rel in chain:
            if rel not in current.children:
                meta = self.get_relation_meta(rel)
                current.children[rel] = SchemaStateNode(rel, meta["dom"], meta["rng"])
            current = current.children[rel]
        current.is_leaf = True

    def get_subgraph_for_inference(self, entity_types):
        subgraph_forest = {}
        for t in entity_types:
            if t in self.roots:
                subgraph_forest[t] = self.roots[t]
        return subgraph_forest if subgraph_forest else None


class SchemaEnhancedTrie(TypeAnchoredTrie):
    def __init__(self, schema_path="freebase_schema.csv"):
        super().__init__(schema_path)


def get_chain(mid, folder_name, name):
    name = normalize_entity_name(name)
    filename = os.path.join(folder_name, f"chains_oneshot_{name}.csv")
    if os.path.exists(filename):
        return
    extractor = RobustOneShotExtractor(SCHEMA_CSV_PATH)
    chains, d1, d2, d3 = extractor.extract_3hop_chains(mid)
    with open(filename, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r1", "r2", "r3"])
        for c in chains:
            w.writerow(c)


def get_schema_trie(mid, folder_name, name):
    name = normalize_entity_name(name)
    trie_json = f"{folder_name}/trie_structure_{name}.json"
    chains_csv = f"{folder_name}/chains_oneshot_{name}.csv"
    if not os.path.exists(chains_csv):
        get_chain(mid, folder_name, name)
    try:
        trie = SchemaEnhancedTrie("freebase_schema.csv")
        trie.build_from_chains(chains_csv)
        return trie
    except Exception as e:
        pass

    if os.path.exists(trie_json) and os.path.exists(chains_csv):
        trie = TypeAnchoredTrie()
        with open(trie_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        new_roots = {}
        for k, v in data.items():
            node = SchemaStateNode(relation=v["rel"], domain=v["schema"]["dom"], range_val=v["schema"]["rng"])
            stack = [(node, v)]
            while stack:
                cur_node, cur_dict = stack.pop()
                for rel, child_dict in cur_dict["children"].items():
                    child = SchemaStateNode(relation=child_dict["rel"], domain=child_dict["schema"]["dom"],
                                            range_val=child_dict["schema"]["rng"])
                    cur_node.children[rel] = child
                    stack.append((child, child_dict))
            new_roots[k] = node
        trie.roots = new_roots
        return trie

    trie = TypeAnchoredTrie()
    trie.build_from_chains(chains_csv)
    serializable_roots = {k: v.to_dict() for k, v in trie.roots.items()}
    with open(trie_json, "w", encoding="utf-8") as f:
        json.dump(serializable_roots, f, indent=2, ensure_ascii=False)
    return trie


def run_relation_plan(question: str, topic_entity: dict, max_tokens=2048, temperature=0.1) -> dict:
    prompt = build_semantic_prompt(question, topic_entity)
    raw = call_llm(MODEL_SET, prompt, max_tokens, temperature)
    json_text = extract_json(raw)
    try:
        data = json.loads(json_text)
    except Exception as e:
        raise ValueError(f"JSON parsing error: {e}\nRaw output:\n{json_text}")
    cleaned = {}
    for ent, arr in data.items():
        new_list = []
        for obj in arr:
            if not isinstance(obj, dict):
                new_list.append({
                    "start_type": [], "path": str(obj), "answer_types": [], "confidence": 0.5, "reasoning": ""
                })
                continue
            start_type = obj.get("start_type", [])
            path = obj.get("path", "")
            answer_types = obj.get("answer_types", [])
            confidence = obj.get("confidence", 0.5)
            reasoning = obj.get("reasoning", "")
            if isinstance(start_type, str): start_type = [start_type]
            if isinstance(answer_types, str): answer_types = [answer_types]
            if not isinstance(confidence, (int, float)): confidence = 0.5
            confidence = max(0.0, min(1.0, float(confidence)))
            new_list.append({
                "start_type": start_type, "path": path, "answer_types": answer_types,
                "confidence": confidence, "reasoning": reasoning
            })
        new_list.sort(key=lambda x: x["confidence"], reverse=True)
        cleaned[ent] = new_list
    return cleaned


def load_descriptions(path="described.jsonl"):
    PRED_LIST_PATH = "pred_list.json"
    PRED_EMB_PATH = "pred_emb.npy"
    RNG_LIST_PATH = "rng_list.json"
    RNG_EMB_PATH = "rng_emb.npy"
    if (os.path.exists(PRED_LIST_PATH) and os.path.exists(PRED_EMB_PATH) and os.path.exists(
            RNG_LIST_PATH) and os.path.exists(RNG_EMB_PATH)):
        pred_items = json.load(open(PRED_LIST_PATH, "r", encoding="utf-8"))
        rng_items = json.load(open(RNG_LIST_PATH, "r", encoding="utf-8"))
        pred_embs = np.load(PRED_EMB_PATH)
        rng_embs = np.load(RNG_EMB_PATH)
        pred2desc = {p: d for (p, d) in pred_items}
        rng2desc = {r: d for (r, d) in rng_items}
        pred2vec = {pred_items[i][0]: pred_embs[i] for i in range(len(pred_items))}
        rng2vec = {rng_items[i][0]: rng_embs[i] for i in range(len(rng_items))}
        return pred2desc, rng2desc, pred2vec, rng2vec
    pred_list = []
    rng_list = []
    pred2desc = {}
    rng2desc = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pred = obj["predicate"]
            rng = obj["range"]
            desc_pred = obj.get("predicate_description", "").strip()
            desc_rng = obj.get("range_description", "").strip()
            if desc_pred:
                pred2desc[pred] = desc_pred
                pred_list.append((pred, desc_pred))
            if desc_rng:
                rng2desc[rng] = desc_rng
                rng_list.append((rng, desc_rng))
    model = get_sentence_model()
    pred_texts = [d for (_, d) in pred_list]
    rng_texts = [d for (_, d) in rng_list]
    pred_emb = model.encode(pred_texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=True)
    rng_emb = model.encode(rng_texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=True)
    json.dump(pred_list, open(PRED_LIST_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(rng_list, open(RNG_LIST_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    np.save(PRED_EMB_PATH, pred_emb)
    np.save(RNG_EMB_PATH, rng_emb)
    pred2vec = {pred_list[i][0]: pred_emb[i] for i in range(len(pred_list))}
    rng2vec = {rng_list[i][0]: rng_emb[i] for i in range(len(rng_list))}
    return pred2desc, rng2desc, pred2vec, rng2vec


def cos_sim(a, b):
    return float(np.dot(a, b))


def embed_sent(text, model):
    return model.encode([text], batch_size=1, normalize_embeddings=True, convert_to_numpy=True)[0]


def score_relation_paths(question, plan_path_str, path_emb, model):
    q_text = question.strip()
    p_text = plan_path_str.strip()
    merged_text = q_text + " " + p_text
    merged_emb = embed_sent(merged_text, model)
    sims = [cos_sim(merged_emb, pe) for pe in path_emb]
    return sims


def score_answer_types(question, answer_types, range_emb, model):
    at_str = ", ".join(answer_types) if answer_types else ""
    merged_text = (question + " " + at_str).strip()
    merged_emb = embed_sent(merged_text, model)
    sims = [cos_sim(merged_emb, re) for re in range_emb]
    return sims


def embed_path_from_predicates(path, pred2vec, dim):
    preds = path[1:]
    vecs = []
    for p in preds:
        if p in pred2vec:
            vecs.append(pred2vec[p])
        else:
            vecs.append(np.zeros(dim, dtype=np.float32))
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)


def embed_range_type(rng, rng2vec, dim):
    return rng2vec.get(rng, np.zeros(dim, dtype=np.float32))


def collect_paths_high_recall(trie, forest, allowed_lengths, answer_types, max_paths=300):
    if allowed_lengths is None:
        allowed_lengths = []
    candidates = []

    def dfs(node, path, type_name):
        hop_len = len(path) - 1

        if hop_len in allowed_lengths:
            end_type = node.range if node.range else "unknown"
            is_compat, at_score = is_type_compatible(end_type, answer_types)
            candidates.append({"path": path[:], "end_type": end_type, "at_score": at_score})

        if hop_len >= max(allowed_lengths):
            return

        for rel, child in node.children.items():
            dfs(child, path + [rel], type_name)

    for type_name, root_node in forest.items():
        dfs(root_node, [type_name], type_name)

    candidates.sort(key=lambda x: x["at_score"], reverse=True)
    candidates = candidates[:max_paths]

    all_paths = [c["path"] for c in candidates]
    all_end_types = [c["end_type"] for c in candidates]
    all_at_scores = [c["at_score"] for c in candidates]
    return all_paths, all_end_types, all_at_scores


def calculate_ted_similarity(path_list, llm_plan_str):
    path_str = " ".join(path_list[1:])

    s1 = path_str.replace(".", " ").replace("_", " ").lower()
    s2 = llm_plan_str.replace("->", " ").replace(".", " ").replace("_", " ").lower()

    # Ratcliff-Obershelp similarity
    ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
    return ratio


def compute_hybrid_scores(all_paths, all_end_types, all_at_scores,
                          sims_paths, sims_ranges,
                          llm_plan_str,
                          plan_confidence=0.5):
    final_scores = []

    for i in range(len(all_paths)):
        path = all_paths[i]

        sp = sims_paths[i]
        sr = sims_ranges[i]
        emb_score = sp + sr
        at_score = all_at_scores[i]
        ted_score = calculate_ted_similarity(path, llm_plan_str)
        base_score = (
                1.0 * emb_score +
                1.0 * ted_score +
                1.0 * at_score
        )

        final_scores.append(base_score * (0.8 + 0.2 * plan_confidence))

    return final_scores


def find_entity_intersections(entity_paths, answer_types, trie):
    if len(entity_paths) < 2: return []
    entity_end_types = {}
    for mid, paths in entity_paths.items():
        end_types = set()
        for path in paths:
            if len(path) >= 2:
                last_rel = path[-1]
                meta = trie.get_relation_meta(last_rel)
                if meta["rng"] != "unknown": end_types.add(meta["rng"])
        entity_end_types[mid] = end_types
    all_types = list(entity_end_types.values())
    if not all_types: return []
    common_types = set.intersection(*all_types) if all(all_types) else set()
    intersections = []
    for common_type in common_types:
        compatible, score = is_type_compatible(common_type, answer_types)
        if compatible or score >= 0.5:
            paths_to_type = {}
            for mid, paths in entity_paths.items():
                for path in paths:
                    if len(path) >= 2:
                        last_rel = path[-1]
                        meta = trie.get_relation_meta(last_rel)
                        if meta["rng"] == common_type:
                            paths_to_type[mid] = path
                            break
            if len(paths_to_type) >= 2:
                intersections.append({"intersection_type": common_type, "paths": paths_to_type, "score": score})
    return intersections


def apply_intersection_boost(global_candidates, intersections, boost_factor=1.5):
    if not intersections: return global_candidates
    intersection_types = {inter["intersection_type"] for inter in intersections}
    intersection_paths = {}
    for inter in intersections:
        for mid, path in inter.get("paths", {}).items():
            intersection_paths[mid] = tuple(path)
    for pid, info in global_candidates.items():
        mid = info.get("mid")
        for cand in info.get("candidates", []):
            if cand.get("end_type") in intersection_types:
                cand["score"] *= boost_factor
            if mid in intersection_paths:
                if tuple(cand.get("path", [])) == intersection_paths[mid]:
                    cand["score"] *= boost_factor
    return global_candidates


def rerank_llm_global(question, final_global_candidates):
    llm_input = {}
    for pid, info in final_global_candidates.items():
        name = info["entity_name"]
        cand_list = info["candidates"]
        options = []
        for idx, c in enumerate(cand_list[:10]):
            options.append({
                "id": idx, "score": round(c["score"], 4),
                "schema_path": " -> ".join(c["path"]), "end_type": c["end_type"]
            })
        llm_input[pid] = {"entity_name": name, "options": options}

    prompt = f"""You are a Freebase schema reasoning expert.
QUESTION: {question}
TASK: Select the TOP 3 best schema paths.
CRITERIA:
1. Semantic Match: Path meaning vs Question.
2. Structure: Logical connection.
3. Answer Type: Matches expectation.
4. Intersection: Common answer type for multi-entity questions.

ENTITIES & PATHS:
{json.dumps(llm_input, indent=2, ensure_ascii=False)}

OUTPUT JSON ONLY:
{{
  "0": [best_id, 2nd_id, 3rd_id],
  ...
}}"""

    resp = call_llm(MODEL_SET, prompt, max_tokens=1024, temperature=0.1)
    try:
        chosen = json.loads(extract_json(resp))
    except:
        chosen = {str(pid): [0] for pid in final_global_candidates.keys()}

    final = {}
    for pid, info in final_global_candidates.items():
        cand_list = info["candidates"]
        name = info["entity_name"]
        mid = info["mid"]
        selected = chosen.get(str(pid), [0])
        if isinstance(selected, int): selected = [selected]
        if not isinstance(selected, list): selected = [0]
        selected = [int(x) for x in selected if isinstance(x, (int, float, str)) and str(x).isdigit()]
        limit = min(len(cand_list), 10)
        selected = [max(0, min(x, limit - 1)) for x in selected]
        if not selected: selected = [0]

        paths_info = []
        for idx in selected:
            c = cand_list[idx]
            paths_info.append({"path": c["path"], "score": c["score"], "end_type": c["end_type"]})
        final[pid] = {"mid": mid, "entity_name": name, "selected_paths": paths_info}
    return final


def penetrate_cvt_node(cvt_mid: str) -> Set[str]:
    names = set()
    probe_query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?p ?o WHERE {{
        ns:{cvt_mid} ?p ?o .
        FILTER(!regex(str(?p), "common\\\\.|freebase\\\\.|kg\\\\.", "i"))
    }} LIMIT 200
    """
    try:
        rows = execurte_sparql(probe_query)
    except:
        return names
    objects = [(r["p"]["value"], r["o"]["value"]) for r in rows if "o" in r]
    for _, o in objects:
        if not o.startswith("http://rdf.freebase.com/ns/"):
            if re.search(r"\d{4}", o): names.add(o)
            continue
        mid = o.replace("http://rdf.freebase.com/ns/", "")
        q1 = f"""
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?name WHERE {{ ns:{mid} ns:type.object.name ?name . FILTER(lang(?name)='en') }} LIMIT 5
        """
        try:
            r1 = execurte_sparql(q1)
            for x in r1:
                nm = x.get("name", {}).get("value")
                if nm: names.add(nm)
        except:
            pass
    return names


def retrieve_entities_by_path_cvt(mid, path, limit=500):
    relations = path[1:]
    hop = len(relations)
    current_entities = {mid}
    start_name = id2entity_name_or_type(mid)
    facts_output = []
    prefix = "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
    MAX_DISPLAY = 30
    for h in range(hop):
        r = relations[h]
        next_entities = set()
        for ent in list(current_entities):
            sparql = prefix + f"SELECT DISTINCT ?o ?name WHERE {{ ns:{ent} ns:{r} ?o . OPTIONAL {{ ?o ns:type.object.name ?name FILTER(lang(?name)='en') }} }} LIMIT {limit}"
            try:
                rows = execurte_sparql(sparql)
            except:
                continue
            for row in rows:
                url = row["o"]["value"]
                if url.startswith("http://rdf.freebase.com/ns/"): next_entities.add(url.split("/")[-1])
        if not next_entities:
            prefix_rel = " -> ".join(relations[:h + 1])
            facts_output.append(f"{start_name} -> {prefix_rel} -> {{ }}")
            current_entities = set()
            continue
        next_names = set()
        cvt_entities = set()
        for e in list(next_entities)[:300]:
            name_e = id2entity_name_or_type(e)
            if name_e != "UnName_Entity" and not name_e.startswith("m.") and not name_e.startswith("g."):
                next_names.add(name_e)
            else:
                cvt_entities.add(e)
        if cvt_entities and len(next_names) < 10:
            for cvt_mid_item in list(cvt_entities)[:10]:
                penetrated_names = penetrate_cvt_node(cvt_mid_item)
                next_names.update(penetrated_names)
                if len(next_names) >= MAX_DISPLAY: break
        prefix_rel = " -> ".join(relations[:h + 1])
        display_names = sorted(next_names)[:MAX_DISPLAY]
        facts_output.append(
            f"{start_name} -> {prefix_rel} -> {{ {', '.join(display_names) if display_names else ''} }}")
        current_entities = next_entities
    return {"facts": facts_output}


def llm_generate_final_answer(question, result):
    prompt = f"""You are a knowledge graph question answering expert.
QUESTION: {question}
FACT CHAINS:
{json.dumps(result, indent=2, ensure_ascii=False)}
INSTRUCTIONS:
1. Analyze question type.
2. Trace chains.
3. Intersection is KEY for multi-entity questions.
4. Ignore empty/irrelevant chains.
5. Output ONLY answer.
ANSWER:"""
    answer = call_llm(MODEL_SET, prompt, max_tokens=1024, temperature=0.1)
    return answer.strip()


def merge_set(global_candidates_paths):
    merged_by_entity = {}
    for pid, info in global_candidates_paths.items():
        mid = info["mid"]
        name = info["entity_name"]
        cand_list = info["candidates"]
        if mid not in merged_by_entity:
            merged_by_entity[mid] = {"entity_name": name, "candidates": {}}
        for c in cand_list:
            key = tuple(c["path"])
            if key not in merged_by_entity[mid]["candidates"]:
                merged_by_entity[mid]["candidates"][key] = c
            else:
                if c["score"] > merged_by_entity[mid]["candidates"][key]["score"]:
                    merged_by_entity[mid]["candidates"][key] = c
    final_global_candidates = {}
    new_pid = 0
    for mid, info in merged_by_entity.items():
        final_global_candidates[new_pid] = {"mid": mid, "entity_name": info["entity_name"],
                                            "candidates": list(info["candidates"].values())}
        new_pid += 1
    return final_global_candidates


def run_question(question, topic_entity, task_id=None):
    pred2desc, rng2desc, pred2vec, rng2vec = get_global_embeddings()
    model = get_sentence_model()

    if task_id is None: task_id = str(uuid.uuid4())[:8]
    folder_name = f"output/{task_id}_{normalize_entity_name(question)[:50]}"

    plan = run_relation_plan(question, topic_entity)
    thread_safe_print(f"[{task_id}][Plan] {plan}")

    plan_count = 0
    global_candidates_paths = {}
    all_entity_paths = {}
    all_answer_types = []

    for mid, name in topic_entity.items():
        thread_safe_print(f"\n[{task_id}][Entity] {name} ({mid})")
        entity_plan = plan.get(name, [])
        if not os.path.exists(folder_name): os.makedirs(folder_name, exist_ok=True)

        trie = get_schema_trie(mid, folder_name, name)
        hop = [1, 2, 3]

        for idx, p in enumerate(entity_plan):
            start_type = p["start_type"]
            path_str = safe_str(p["path"])
            answer_types = p["answer_types"]
            plan_confidence = p.get("confidence", 0.5)
            all_answer_types.extend(answer_types)

            forest = trie.get_subgraph_for_inference(start_type)
            if forest is None: continue

            all_paths, all_end_types, all_at_scores = collect_paths_high_recall(
                trie, forest, hop, answer_types, max_paths=300
            )

            if not all_paths: continue

            dim = next(iter(pred2vec.values())).shape[0]
            path_emb = np.array([embed_path_from_predicates(pth, pred2vec, dim) for pth in all_paths])
            range_emb = np.array([embed_range_type(t, rng2vec, dim) for t in all_end_types])
            sims_paths = score_relation_paths(question, path_str, path_emb, model)
            sims_ranges = score_answer_types(question, answer_types, range_emb, model)

            final_scores = compute_hybrid_scores(
                all_paths, all_end_types, all_at_scores,
                sims_paths, sims_ranges,
                llm_plan_str=path_str,
                plan_confidence=plan_confidence
            )

            ranked = sorted(zip(final_scores, all_paths, all_end_types), key=lambda x: x[0], reverse=True)
            candidate_list = []
            for score, pth, endt in ranked[:20]:
                candidate_list.append({"score": float(score), "path": pth, "end_type": endt})

            global_candidates_paths[plan_count] = {
                "mid": mid, "entity_name": name, "candidates": candidate_list
            }
            plan_count += 1
            if mid not in all_entity_paths: all_entity_paths[mid] = []
            all_entity_paths[mid].extend([c["path"] for c in candidate_list])

    if len(topic_entity) > 1 and all_entity_paths:
        first_mid = list(topic_entity.keys())[0]
        first_name = list(topic_entity.values())[0]
        trie = get_schema_trie(first_mid, folder_name, first_name)
        unique_answer_types = list(set(all_answer_types))
        intersections = find_entity_intersections(all_entity_paths, unique_answer_types, trie)
        if intersections:
            global_candidates_paths = apply_intersection_boost(global_candidates_paths, intersections)

    if not global_candidates_paths:
        thread_safe_print(f"[{task_id}][Warning] No candidates collected")
        return ""

    global_candidates_paths = merge_set(global_candidates_paths)
    best_paths_plan = rerank_llm_global(question, global_candidates_paths)

    result = {}
    for pid, best_info in best_paths_plan.items():
        mid = best_info["mid"]
        entity_name = best_info["entity_name"]
        selected_paths = best_info["selected_paths"]
        all_facts = []
        for path_info in selected_paths:
            best_path = path_info["path"]
            retrieved = retrieve_entities_by_path_cvt(mid, best_path)
            all_facts.extend(retrieved["facts"])
        seen = set()
        unique_facts = []
        for f in all_facts:
            if f not in seen:
                seen.add(f)
                unique_facts.append(f)
        result[pid] = {"entity": entity_name, "facts": unique_facts}

    final_answer = llm_generate_final_answer(question, result)
    thread_safe_print(f"[{task_id}][Answer] {final_answer}")
    return final_answer


def exact_match(response, answers):
    clean_result = response.strip().replace(" ", "").replace('-', '').lower()
    for answer in answers:
        clean_answer = answer.strip().replace(" ", "").replace('-', '').lower()
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return True
    return False


def evaluate_single_item(args):
    idx, item = args
    question = item["question"]
    topic_entity = item["topic_entity"]
    gold = [item["answer"]]
    task_id = f"T{idx:04d}"
    try:
        pred = run_question(question, topic_entity, task_id=task_id)
        correct = exact_match(pred, gold)
        f1, _, _ = eval_f1([pred], gold)
        return {"idx": idx, "question": question, "gold": gold, "pred": pred, "correct": correct, "f1": f1}
    except Exception as e:
        return {"idx": idx, "question": question, "gold": gold, "pred": f"ERROR: {e}", "correct": False, "f1": 0.0}


def evaluate_parallel(cwq_path, limit=100, workers=8):
    print("[Init] Loading model and embeddings...")
    _ = get_sentence_model()
    _ = get_global_embeddings()
    _ = load_schema_types()
    print("[Init] Done. Starting evaluation...")
    with open(cwq_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data[:limit]
    hits1 = 0
    results = [None] * len(data)
    completed = 0
    total_f1 = 0.0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(evaluate_single_item, (i, item)): i for i, item in enumerate(data)}
        pbar = tqdm(total=len(data), desc="Parallel CWQ")
        for future in as_completed(future_to_idx):
            res = future.result()
            idx = res["idx"]
            results[idx] = res
            if res["correct"]: hits1 += 1
            total_f1 += res.get("f1", 0.0)
            completed += 1
            acc = hits1 / completed
            avg_f1 = total_f1 / completed
            pbar.update(1)
            pbar.set_postfix(
                {"acc": f"{acc:.4f}", "f1": f"{avg_f1:.4f}", "pred": res["pred"][:50] if res["pred"] else ""})
        pbar.close()
    print("\n=== Evaluation Complete ===")
    print(f"Total = {limit}, Hits@1 = {hits1}, Accuracy = {hits1 / limit:.4f}")
    print(f"Average F1 = {total_f1 / limit:.4f}")
    with open("cwq_predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return hits1 / limit


if __name__ == "__main__":
    evaluate_parallel("data/cwq.json", limit=100, workers=8)