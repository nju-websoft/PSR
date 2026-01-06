from SPARQLWrapper import SPARQLWrapper, JSON
import os

SPARQLPATH = "http://localhost:8890/sparql"
SPARQL_CONFIG = {
    "SPARQLPATH": "http://localhost:8890/sparql",
    "sparql_head_relations": """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation
WHERE {
  ns:%s ?relation ?x .
}""",
    "sparql_tail_relations": """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation
WHERE {
  ?x ?relation ns:%s .
}""",
    "sparql_tail_entities_extract": """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?tailEntity
WHERE {
  ns:%s ns:%s ?tailEntity .
}""",
    "sparql_head_entities_extract": """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?tailEntity
WHERE {
  ?tailEntity ns:%s ns:%s .
}""",
    "sparql_id": """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?tailEntity
WHERE {
  {
    ?entity ns:type.object.name ?tailEntity .
    FILTER(?entity = ns:%s)
  }
  UNION
  {
    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .
    FILTER(?entity = ns:%s)
  }
}"""
}
SPARQLPATH = SPARQL_CONFIG["SPARQLPATH"]
sparql_head_relations = SPARQL_CONFIG["sparql_head_relations"]
sparql_tail_relations = SPARQL_CONFIG["sparql_tail_relations"]
sparql_tail_entities_extract = SPARQL_CONFIG["sparql_tail_entities_extract"]
sparql_head_entities_extract = SPARQL_CONFIG["sparql_head_entities_extract"]
sparql_id = SPARQL_CONFIG["sparql_id"]


def abandon_rels(relation):
    if not relation or not isinstance(relation, str):
        return True
    if (
            relation == "type.object.type"
            or relation == "type.object.name"
            or relation.startswith("common.")
            or relation.startswith("freebase.")
            or "sameAs" in relation
    ):
        return True
    if relation.startswith(("http://", "https://")):
        return True
    if ".key/" in relation or ".key." in relation:
        return True
    if relation.startswith(("type.type.", "base.", "kg.")):
        return True
    if relation in ("type.type.instance", "type.object.key", "rdfs:label", "rdf:type"):
        return True
    return False


def execurte_sparql(sparql_query):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]


def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/", "") for relation in relations]


def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/", "") for entity in entities]


def id2entity_name_or_type(mid):
    def run(q):
        try:
            sp = SPARQLWrapper(SPARQLPATH)
            sp.setQuery(q)
            sp.setReturnFormat(JSON)
            res = sp.query().convert()
            return res.get("results", {}).get("bindings", [])
        except:
            return []

    q1 = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?n WHERE {{
        ns:{mid} ns:type.object.name ?n .
        FILTER (lang(?n) = 'en')
    }} LIMIT 1
    """
    rows = run(q1)
    if rows:
        return rows[0]["n"]["value"]
    q2 = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?a WHERE {{
        ns:{mid} ns:common.topic.alias ?a .
        FILTER (lang(?a) = 'en')
    }} LIMIT 1
    """
    rows = run(q2)
    if rows:
        return rows[0]["a"]["value"]
    return "UnName_Entity"


sparql_types_dual = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?type WHERE {
  { ns:%s rdf:type ?type . }
  UNION
  { ns:%s ns:type.object.type ?type . }
}"""

sparql_head_relations_3hop = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation1 ?relation2 ?relation3\nWHERE {\n  ns:%s ?relation1 ?intermediate1 .\n  ?intermediate1 ?relation2 ?intermediate2 .\n  ?intermediate2 ?relation3 ?x .\n}"""


def get_entity_types(entity_id):
    sparql_query = sparql_types_dual % (entity_id, entity_id)
    results = execurte_sparql(sparql_query)
    types = [r["type"]["value"].replace("http://rdf.freebase.com/ns/", "") for r in results]
    types = list(set(types))
    filtered_types = [t for t in types if not t.startswith(("common.", "base.", "freebase.", "kg."))]
    return filtered_types or types


def replace_relation_prefix_2hop(relations):
    return [(relation['relation1']['value'].replace("http://rdf.freebase.com/ns/", ""),
             relation['relation2']['value'].replace("http://rdf.freebase.com/ns/", "")) for relation in relations]


def replace_relation_prefix_3hop(relations):
    return [(relation['relation1']['value'].replace("http://rdf.freebase.com/ns/", ""),
             relation['relation2']['value'].replace("http://rdf.freebase.com/ns/", ""),
             relation['relation3']['value'].replace("http://rdf.freebase.com/ns/", "")) for relation in relations]


from SPARQLWrapper import SPARQLWrapper, JSON


def generate_and_run_sparql(mid, path, folder_name):
    if len(path) > 6:
        return []

    print(f"\nStep 5: Executing SPARQL for {mid} (Path Len: {len(path)})...")

    if not path:
        return []
    select_vars = []
    for i in range(len(path)):
        select_vars.append(f"?x{i + 1}")
        select_vars.append(f"?x{i + 1}_name")

    sparql_query = (
            "PREFIX ns: <http://rdf.freebase.com/ns/>\n"
            "SELECT DISTINCT " + " ".join(select_vars) + " WHERE {\n"
    )

    current_s = f"ns:{mid}"
    for i, relation in enumerate(path):
        next_o = f"?x{i + 1}"
        name_var = f"?x{i + 1}_name"
        if relation.startswith("http://") or relation.startswith("https://"):
            pred = f"<{relation}>"
        else:
            pred = f"ns:{relation}"
        sparql_query += f"  {current_s} {pred} {next_o} .\n"
        sparql_query += (
            f"  OPTIONAL {{ {next_o} ns:type.object.name {name_var} "
            f"FILTER(lang({name_var}) = 'en') }} .\n"
        )
        current_s = next_o

    sparql_query += "} LIMIT 500"
    with open(f"{folder_name}/sparql_L{len(path)}.txt", "w", encoding="utf-8") as f:
        f.write(sparql_query)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setMethod("POST")
    sparql.setReturnFormat(JSON)
    sparql.setQuery(sparql_query)
    results = []
    try:
        data = sparql.query().convert()
        results = data.get("results", {}).get("bindings", [])
    except Exception as e:
        print(f"SPARQL Error: {e}")
        return []
    needs_extension = False
    sample_cvt_mid = None

    if results:
        last_idx = len(path)
        last_name_key = f"x{last_idx}_name"
        last_id_key = f"x{last_idx}"
        first_row = results[0]
        has_name = first_row.get(last_name_key, {}).get("value")
        if not has_name:
            raw_id = first_row.get(last_id_key, {}).get("value", "")
            if raw_id:
                sample_cvt_mid = raw_id.split("/")[-1]
                needs_extension = True

    if needs_extension:
        print(f"  [Auto-Fix] End-node {sample_cvt_mid} has no name. Attempting CVT penetration...")
        probe_query = f"""
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?p WHERE {{
            ns:{sample_cvt_mid} ?p ?o .
            ?o ns:type.object.name ?n .
            FILTER (isIRI(?o))
            FILTER (!regex(str(?p), "common\\\\.|type\\\\.|base\\\\.|freebase\\\\.|kg\\\\.", "i"))
            FILTER (!regex(str(?p), "w3\\\\.org", "i")) 
        }} LIMIT 1
        """
        try:
            probe_sparql = SPARQLWrapper(SPARQLPATH)
            probe_sparql.setQuery(probe_query)
            probe_sparql.setReturnFormat(JSON)
            probe_res = probe_sparql.query().convert()
            probe_bindings = probe_res.get("results", {}).get("bindings", [])

            if probe_bindings:
                new_rel = probe_bindings[0]["p"]["value"].replace("http://rdf.freebase.com/ns/", "")
                print(f"  [Auto-Fix] Extending path with: {new_rel}")
                return generate_and_run_sparql(mid, path + [new_rel], folder_name)
        except Exception as e:
            print(f"  [Auto-Fix] Probe failed: {e}")
    final_answers = []
    for r in results:
        chain_str = []
        for i in range(len(path)):
            name_val = r.get(f"x{i + 1}_name", {}).get("value")
            if not name_val:
                raw_val = r.get(f"x{i + 1}", {}).get("value", "")
                mid_val = raw_val.split("/")[-1]
                name_val = id2entity_name_or_type(mid_val)

            chain_str.append(name_val)
        final_answers.append(" -> ".join(chain_str))
    with open(f"{folder_name}/final_answer.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(final_answers))

    print(f"final_answers sample: {final_answers[:3]}")
    return final_answers


def retrieve_entities_by_path(mid, path, limit=500):
    if not path or len(path) < 2:
        return {"facts": []}

    relations = path[1:]
    hop = len(relations)

    current_entities = {mid}
    start_name = id2entity_name_or_type(mid)
    facts_output = []

    prefix = "PREFIX ns: <http://rdf.freebase.com/ns/>\n"

    for h in range(hop):

        r = relations[h]
        next_entities = set()
        for ent in current_entities:

            sparql = (
                    prefix +
                    "SELECT DISTINCT ?o WHERE {\n"
                    f"  ns:{ent} ns:{r} ?o .\n"
                    "} LIMIT %d" % limit
            )

            try:
                rows = execurte_sparql(sparql)
            except:
                continue

            for row in rows:
                url = row["o"]["value"]
                obj = url.split("/")[-1]
                next_entities.add(obj)
        if not next_entities:
            prefix_rel = " -> ".join(relations[:h + 1])
            facts_output.append(
                f"{start_name} -> {prefix_rel} -> {{ }}"
            )
            current_entities = set()
            continue
        next_names = {
            id2entity_name_or_type(e)
            for e in next_entities
            if id2entity_name_or_type(e) != "UnName_Entity"
        }
        prefix_rel = " -> ".join(relations[:h + 1])
        facts_output.append(
            f"{start_name} -> {prefix_rel} -> {{ " + ", ".join(sorted(next_names)) + " }}"
        )
        current_entities = next_entities

    return {"facts": facts_output}


def replace_relation_prefix_entity(entities, hop=2):
    if hop == 3:
        return [(entity['entity1']['value'].replace("http://rdf.freebase.com/ns/", ""),
                 entity['entity2']['value'].replace("http://rdf.freebase.com/ns/", ""),
                 entity['entity3']['value'].replace("http://rdf.freebase.com/ns/", "")) for entity in entities]
    elif hop == 2:
        return [(entity['entity1']['value'].replace("http://rdf.freebase.com/ns/", ""),
                 entity['entity2']['value'].replace("http://rdf.freebase.com/ns/", "")) for entity in entities]


def retrieve_entity_sparql(entity, relation_1, relation_2, relation_3):
    sparql_retrieve_entity = '''
PREFIX ns: <http://rdf.freebase.com/ns/>

SELECT DISTINCT ?entity1 ?entity2 ?entity3
WHERE {
  ns:%s ns:%s ?entity1 .
  ?entity1 ns:%s ?entity2 .
  ?entity2 ns:%s ?entity3 .
}
LIMIT 500
'''

    sparql_retrieve_entity_query = sparql_retrieve_entity % (entity, relation_1, relation_2, relation_3)

    try:
        retrieved_entities = execurte_sparql(sparql_retrieve_entity_query)
        retrieved_entities = replace_relation_prefix_entity(retrieved_entities, hop=3)
        retrieved_entities = [
            [id2entity_name_or_type(entity1), id2entity_name_or_type(entity2), id2entity_name_or_type(entity3)] for
            entity1, entity2, entity3 in retrieved_entities]
        return retrieved_entities
    except Exception as e:
        print(f"An error occurred: {e}")
        return []