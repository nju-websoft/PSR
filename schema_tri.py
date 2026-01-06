import csv
import json
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional


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
        count = 0
        with open(chain_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                chain = [r.strip() for r in row]
                self._insert_chain_smart(chain)
                count += 1

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
        if not subgraph_forest:
            return None
        return subgraph_forest

    def debug_print_node(self, node, level=0):
        indent = "  " * level
        info = f"[{node.relation}] ({node.domain}->{node.range})" if node.relation != "ROOT_TYPE" else "[ROOT]"
        print(f"{indent}{info}")
        for child in node.children.values():
            self.debug_print_node(child, level + 1)


class EnhancedSchemaStateNode:

    def __init__(self, relation=None, domain=None, range_val=None):
        self.relation = relation
        self.domain = domain
        self.range = range_val
        self.children = {}
        self.is_leaf = False
        self.parents = []
        self.reachable_from_types = set()

    def to_dict(self):
        return {
            "rel": self.relation,
            "schema": {"dom": self.domain, "rng": self.range},
            "children": {k: v.to_dict() for k, v in self.children.items()}
        }


class SchemaEnhancedTrie:

    def __init__(self, schema_path="freebase_schema.csv"):
        self.pred_meta = {}
        self.type_to_predicates = defaultdict(set)
        self.range_to_predicates = defaultdict(set)
        self.roots = defaultdict(lambda: EnhancedSchemaStateNode(relation="ROOT_TYPE"))
        self._load_schema(schema_path)

    def _load_schema(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        dom, pred, rng = row[0].strip(), row[1].strip(), row[2].strip()
                        self.pred_meta[pred] = {"dom": dom, "rng": rng}
                        self.type_to_predicates[dom].add(pred)
                        self.range_to_predicates[rng].add(pred)
        except Exception as e:
            print(f"Schema Load Error: {e}")

    def get_relation_meta(self, relation):
        return self.pred_meta.get(relation, {"dom": "unknown", "rng": "unknown"})

    def validate_path_type_consistency(self, path: List[str], start_types: List[str]) -> Tuple[bool, float, List[str]]:

        if len(path) < 2:
            return False, 0.0, []

        relations = path[1:]
        current_types = set(start_types)
        type_chain = [set(start_types)]
        valid_hops = 0
        total_hops = len(relations)

        for rel in relations:
            meta = self.get_relation_meta(rel)
            if meta["dom"] == "unknown":
                type_chain.append({"unknown"})
                valid_hops += 0.5
                current_types = {"unknown"}
                continue

            expected_domain = meta["dom"]
            domain_match = self._type_compatible(current_types, expected_domain)

            if domain_match:
                valid_hops += 1
            else:
                valid_hops += 0.3

            current_types = {meta["rng"]} if meta["rng"] != "unknown" else {"unknown"}
            type_chain.append(current_types)

        consistency_score = valid_hops / total_hops if total_hops > 0 else 0.0
        is_valid = consistency_score >= 0.7

        return is_valid, consistency_score, [list(t) for t in type_chain]

    def _type_compatible(self, current_types: Set[str], expected_domain: str) -> bool:
        if "unknown" in current_types:
            return True
        if expected_domain in current_types:
            return True
        expected_prefix = expected_domain.split('.')[0] if '.' in expected_domain else expected_domain
        for t in current_types:
            t_prefix = t.split('.')[0] if '.' in t else t
            if t_prefix == expected_prefix:
                return True
        return False

    def filter_paths_by_answer_type(self, paths: List[List[str]],
                                    answer_types: List[str],
                                    tolerance: float = 0.5) -> List[Tuple[List[str], float]]:
        if not answer_types:
            return [(p, 1.0) for p in paths]

        result = []
        answer_type_set = set(answer_types)
        answer_prefixes = {t.split('.')[0] for t in answer_types if '.' in t}

        for path in paths:
            if len(path) < 2:
                continue
            last_rel = path[-1]
            meta = self.get_relation_meta(last_rel)
            end_type = meta["rng"]

            if end_type in answer_type_set:
                score = 1.0
            elif end_type != "unknown":
                end_prefix = end_type.split('.')[0] if '.' in end_type else end_type
                if end_prefix in answer_prefixes:
                    score = 0.8
                else:
                    score = tolerance
            else:
                score = tolerance

            result.append((path, score))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def collect_paths_with_type_constraints(
            self,
            forest: Dict[str, EnhancedSchemaStateNode],
            allowed_lengths: List[int],
            answer_types: Optional[List[str]] = None,
            max_paths: int = 100
    ) -> List[Dict]:

        candidates = []
        answer_type_set = set(answer_types) if answer_types else set()
        answer_prefixes = {t.split('.')[0] for t in answer_types if answer_types and '.' in t}

        def dfs(node, path, type_chain):
            hop_len = len(path) - 1

            if hop_len in allowed_lengths and node.relation != "ROOT_TYPE":
                end_type = node.range or "unknown"

                if end_type in answer_type_set:
                    at_score = 1.0
                elif end_type != "unknown" and end_type.split('.')[0] in answer_prefixes:
                    at_score = 0.8
                else:
                    at_score = 0.3

                candidates.append({
                    "path": path[:],
                    "end_type": end_type,
                    "type_chain": type_chain[:],
                    "consistency_score": 1.0,
                    "answer_type_score": at_score
                })

            for rel, child in node.children.items():
                new_type_chain = type_chain + [child.range or "unknown"]
                dfs(child, path + [rel], new_type_chain)

        for type_name, root_node in forest.items():
            dfs(root_node, [type_name], [type_name])

        candidates.sort(key=lambda x: x["answer_type_score"], reverse=True)
        return candidates[:max_paths]

    def validate_llm_generated_path(self, path_str: str, start_types: List[str]) -> Dict:
        parts = [p.strip() for p in path_str.split('->')]
        if len(parts) < 2:
            return {"is_valid": False, "error": "path too short"}

        relations = parts[1:]
        missing = []
        violations = []
        fixes = []
        current_types = set(start_types)

        for i, rel in enumerate(relations):
            meta = self.get_relation_meta(rel)
            if meta["dom"] == "unknown":
                missing.append(rel)
                similar = self._find_similar_relation(rel)
                if similar:
                    fixes.append(f"Replace '{rel}' with '{similar}'")
                continue

            if not self._type_compatible(current_types, meta["dom"]):
                violations.append({
                    "hop": i,
                    "relation": rel,
                    "expected_domain": meta["dom"],
                    "actual_types": list(current_types)
                })

            current_types = {meta["rng"]}

        is_valid = len(missing) == 0 and len(violations) == 0
        consistency = 1.0 - (len(missing) + len(violations)) / max(len(relations), 1)

        return {
            "is_valid": is_valid,
            "consistency_score": max(0, consistency),
            "missing_relations": missing,
            "type_violations": violations,
            "suggested_fixes": fixes
        }

    def _find_similar_relation(self, rel: str, threshold: float = 0.7) -> Optional[str]:
        if '.' not in rel:
            return None
        domain_prefix = rel.rsplit('.', 1)[0]
        candidates = [p for p in self.pred_meta.keys() if p.startswith(domain_prefix)]
        if candidates:
            return min(candidates, key=lambda x: self._edit_distance(x, rel))
        return None

    def _edit_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    def find_intersection_paths(
            self,
            entity_paths: Dict[str, List[List[str]]],
            answer_types: List[str]
    ) -> List[Dict]:
        if len(entity_paths) < 2:
            return []

        entity_end_types = {}
        for mid, paths in entity_paths.items():
            end_types = set()
            for path in paths:
                if len(path) >= 2:
                    last_rel = path[-1]
                    meta = self.get_relation_meta(last_rel)
                    if meta["rng"] != "unknown":
                        end_types.add(meta["rng"])
            entity_end_types[mid] = end_types

        all_types = list(entity_end_types.values())
        common_types = set.intersection(*all_types) if all_types and all(all_types) else set()

        intersection_candidates = []

        for common_type in common_types:
            at_match = common_type in set(answer_types)
            prefix_match = any(
                common_type.split('.')[0] == at.split('.')[0]
                for at in answer_types if '.' in at
            )

            if at_match or prefix_match:
                paths_to_type = {}
                for mid, paths in entity_paths.items():
                    for path in paths:
                        if len(path) >= 2:
                            last_rel = path[-1]
                            meta = self.get_relation_meta(last_rel)
                            if meta["rng"] == common_type:
                                paths_to_type[mid] = path
                                break

                if len(paths_to_type) == len(entity_paths):
                    intersection_candidates.append({
                        "intersection_type": common_type,
                        "paths": paths_to_type,
                        "score": 1.0 if at_match else 0.8
                    })

        return intersection_candidates

    def build_from_chains(self, chain_path: str):

        count = 0
        with open(chain_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                chain = [r.strip() for r in row]
                self._insert_chain_smart(chain)
                count += 1

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
                child = EnhancedSchemaStateNode(rel, meta["dom"], meta["rng"])
                child.parents.append(current)
                current.children[rel] = child
            current = current.children[rel]
        current.is_leaf = True

    def get_subgraph_for_inference(self, entity_types: List[str]) -> Optional[Dict]:
        subgraph_forest = {}
        for t in entity_types:
            if t in self.roots:
                subgraph_forest[t] = self.roots[t]
        return subgraph_forest if subgraph_forest else None

    def debug_print_node(self, node, level=0):
        indent = "  " * level
        info = f"[{node.relation}] ({node.domain}->{node.range})" if node.relation != "ROOT_TYPE" else "[ROOT]"
        print(f"{indent}{info}")
        for child in node.children.values():
            self.debug_print_node(child, level + 1)


def compute_schema_enhanced_score(
        path: List[str],
        start_types: List[str],
        answer_types: List[str],
        trie: SchemaEnhancedTrie,
        embedding_score: float,
        weights: Dict[str, float] = None
) -> float:
    if weights is None:
        weights = {
            "embedding": 0.4,
            "type_consistency": 0.3,
            "answer_type_match": 0.3
        }

    is_valid, consistency_score, _ = trie.validate_path_type_consistency(path, start_types)
    if len(path) >= 2:
        last_rel = path[-1]
        meta = trie.get_relation_meta(last_rel)
        end_type = meta["rng"]

        if end_type in set(answer_types):
            at_score = 1.0
        elif end_type != "unknown":
            end_prefix = end_type.split('.')[0]
            answer_prefixes = {t.split('.')[0] for t in answer_types if '.' in t}
            at_score = 0.7 if end_prefix in answer_prefixes else 0.3
        else:
            at_score = 0.3
    else:
        at_score = 0.0

    final_score = (
            weights["embedding"] * embedding_score +
            weights["type_consistency"] * consistency_score +
            weights["answer_type_match"] * at_score
    )

    return final_score


def rerank_with_schema_constraints(
        candidates: List[Dict],
        trie: SchemaEnhancedTrie,
        start_types: List[str],
        answer_types: List[str]
) -> List[Dict]:
    enhanced_candidates = []

    for cand in candidates:
        path = cand["path"]
        original_score = cand["score"]

        is_valid, consistency_score, type_chain = trie.validate_path_type_consistency(
            path, start_types
        )

        end_type = cand.get("end_type", "unknown")
        if end_type in set(answer_types):
            at_score = 1.0
        elif end_type != "unknown":
            end_prefix = end_type.split('.')[0]
            answer_prefixes = {t.split('.')[0] for t in answer_types if '.' in t}
            at_score = 0.7 if end_prefix in answer_prefixes else 0.3
        else:
            at_score = 0.3

        enhanced_score = (
                0.4 * original_score +
                0.3 * consistency_score +
                0.3 * at_score
        )

        enhanced_candidates.append({
            **cand,
            "original_score": original_score,
            "enhanced_score": enhanced_score,
            "consistency_score": consistency_score,
            "answer_type_score": at_score,
            "type_chain": type_chain,
            "is_type_valid": is_valid
        })

    enhanced_candidates.sort(key=lambda x: x["enhanced_score"], reverse=True)
    return enhanced_candidates