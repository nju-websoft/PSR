# -*- coding: utf-8 -*-
import csv
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQLPATH = "http://localhost:8890/sparql"
SCHEMA_CSV_PATH = "freebase_schema.csv"
TIMEOUT_SEC = 300


BAD_DOMAINS = [
    "common", "type", "kg", "base", "user",
    "dataworld", "key", "wikipedia", "kp"
]


class RobustOneShotExtractor:
    def __init__(self, schema_path):
        self.sparql = SPARQLWrapper(SPARQLPATH)
        self.sparql.setMethod("POST")
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(TIMEOUT_SEC)
        self.valid_predicates = self._load_schema_whitelist(schema_path)

    def _load_schema_whitelist(self, path):
        valid_preds = set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        valid_preds.add(row[1].strip())
        except:
            pass
        return valid_preds

    def extract_3hop_chains(self, entity_id):
        print(f"Starting 3-Hop Chain Search for {entity_id}...")
        bad_regex_parts = [f"/{d}[.]" for d in BAD_DOMAINS]
        regex_pattern = "(" + "|".join(bad_regex_parts) + ")"

        query = """
        PREFIX ns: <http://rdf.freebase.com/ns/>

        SELECT DISTINCT ?r1 ?r2 ?r3
        WHERE {
          # Hop 1
          ns:%s ?r1 ?m1 .
          FILTER (isIRI(?m1)) 
          FILTER (!regex(str(?r1), "%s", "i"))

          # Hop 2
          ?m1 ?r2 ?m2 .
          FILTER (isIRI(?m2))
          FILTER (!regex(str(?r2), "%s", "i"))
          # Hop 3
          ?m2 ?r3 ?x .
          FILTER (!regex(str(?r3), "%s", "i"))
        }
        """ % (entity_id, regex_pattern, regex_pattern, regex_pattern)

        self.sparql.setQuery(query)

        valid_chains = []
        try:
            print("  -> Sending query to Virtuoso...")
            results = self.sparql.query().convert()["results"]["bindings"]
            print(f"  -> Database returned {len(results)} raw chains.")

            for r in results:
                r1 = r["r1"]["value"].replace("http://rdf.freebase.com/ns/", "")
                r2 = r["r2"]["value"].replace("http://rdf.freebase.com/ns/", "")
                r3 = r["r3"]["value"].replace("http://rdf.freebase.com/ns/", "")

                if self.valid_predicates:
                    if (r1 in self.valid_predicates and
                            r2 in self.valid_predicates and
                            r3 in self.valid_predicates):
                        valid_chains.append((r1, r2, r3))
                else:
                    valid_chains.append((r1, r2, r3))

        except Exception as e:
            print(f"SPARQL Error: {e}")
            return [], [], [], []

        print(f"  -> Final Valid Chains: {len(valid_chains)}")

        d1 = list(set([c[0].split('.')[0] + '.' + c[0].split('.')[1] for c in valid_chains if '.' in c[0]]))
        d2 = list(set([c[1].split('.')[0] + '.' + c[1].split('.')[1] for c in valid_chains if '.' in c[1]]))
        d3 = list(set([c[2].split('.')[0] + '.' + c[2].split('.')[1] for c in valid_chains if '.' in c[2]]))

        return valid_chains, d1, d2, d3
