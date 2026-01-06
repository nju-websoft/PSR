import json
import re
from freebase_func import get_entity_types


def extract_json(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"```$", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0).strip() if m else text


INVALID_PREFIXES = (
    "common.", "base.", "user.", "freebase.", "type.",
    "schema.", "dataworld."
)
INVALID_EXACT = {
    "common.topic", "common.document", "common.webpage"
}
INVALID_SUFFIX = (
    "topic", "object", "entity"
)


def filter_valid_types(types):
    valid = []
    for t in types:
        if not isinstance(t, str):
            continue
        if t.startswith(INVALID_PREFIXES):
            continue
        if "." not in t:
            continue
        if t in INVALID_EXACT:
            continue
        if any(t.endswith(s) for s in INVALID_SUFFIX):
            continue
        valid.append(t)
    return valid


RELATION_PLAN_PROMPT = r"""
Given a question and some Topic Entities, each Topic Entity has a list of Freebase types.
You must generate possible Freebase relation paths for each Topic Entity.

Here are the RULES you must obey:
1. Output a JSON dict: keys = topic entity surface names.
2. For each topic entity output a list of **path objects**.
3. Each path object MUST contain:
   - "start_type": a list of Freebase types **chosen ONLY** from the given Topic Entity Types
   - "path": a relation chain "Entity -> r1 -> r2 -> ..."
   - "answer_types": plausible Freebase types of the final answer entity
   - "confidence": float 0.0-1.0, how confident you are this path answers the question
   - "reasoning": brief explanation of why this path semantically fits the question
4. For each entity output at least 3 path objects, ranked by confidence (highest first).
5. NEVER invent types. Use only the provided clean types.
6. Think carefully about:
   - What is the question really asking?
   - What constraints connect the entities?
   - What type of answer is expected?

# Example 1
Question: Find the person who said "Taste cannot be controlled by law", where did this person die from?
Topic Entities: ["\"Taste cannot be controlled by law\""]
Topic Entities Types: [["media_common.quotation"]]
Output: {
"\"Taste cannot be controlled by law\"":[
    {
      "start_type":["media_common.quotation"],
      "path":"\"Taste cannot be controlled by law\" -> media_common.quotation.author -> people.deceased_person.place_of_death",
      "answer_types":["location.location"],
    },
    {
      "start_type":["media_common.quotation"],
      "path":"\"Taste cannot be controlled by law\" -> media_common.quotation.author -> people.person.place_of_death",
      "answer_types":["location.location"],
    },
    {
      "start_type":["media_common.quotation"],
      "path":"\"Taste cannot be controlled by law\" -> media_common.quotation.source -> book.written_work.author -> people.deceased_person.place_of_death",
      "answer_types":["location.location"],
    }
]
}

# Example 2
Question: Who is the director of the movie featured Miley Cyrus and was produced by Tobin Armbrust?
Topic Entities: ["Miley Cyrus","Tobin Armbrust"]
Topic Entities Types: [
  ["people.person","film.actor"],
  ["people.person","film.producer"]
]
Output: {
"Miley Cyrus":[
    {
      "start_type":["film.actor"],
      "path":"Miley Cyrus -> film.actor.film -> film.performance.film -> film.film.directed_by",
      "answer_types":["film.director","people.person"],
    },
    {
      "start_type":["film.actor"],
      "path":"Miley Cyrus -> film.actor.film -> film.performance.film -> film.film.director",
      "answer_types":["film.director","people.person"],
    },
    {
      "start_type":["people.person"],
      "path":"Miley Cyrus -> film.person_or_entity_appearing_in_film.films -> film.personal_film_appearance.film -> film.film.directed_by",
      "answer_types":["film.director"],
    }
],
"Tobin Armbrust":[
    {
      "start_type":["film.producer"],
      "path":"Tobin Armbrust -> film.producer.film -> film.film.directed_by",
      "answer_types":["film.director","people.person"],
    },
    {
      "start_type":["film.producer"],
      "path":"Tobin Armbrust -> film.producer.films_executive_produced -> film.film.directed_by",
      "answer_types":["film.director","people.person"],
    },
    {
      "start_type":["people.person"],
      "path":"Tobin Armbrust -> film.film.executive_produced_by -> film.film.directed_by",
      "answer_types":["film.director"],
    }
]
}

# Example 3
Question: What major religion in the UK has a place of worship named St. Mary's Cathedral, Batticaloa?
Topic Entities:["United Kingdom","St. Mary's Cathedral, Batticaloa"]
Topic Entities Types: [
  ["location.country","location.statistical_region"],
  ["religion.place_of_worship","location.location"]
]
Output: {
"United Kingdom":[
    {
      "start_type":["location.statistical_region"],
      "path":"United Kingdom -> location.statistical_region.religions -> location.religion_percentage.religion",
      "answer_types":["religion.religion"],
    },
    {
      "start_type":["location.country"],
      "path":"United Kingdom -> location.country.official_religion",
      "answer_types":["religion.religion"],
    },
    {
      "start_type":["location.country"],
      "path":"United Kingdom -> location.country.religions_practiced -> religion.religion",
      "answer_types":["religion.religion"],
    }
],
"St. Mary's Cathedral, Batticaloa":[
    {
      "start_type":["religion.place_of_worship"],
      "path":"St. Mary's Cathedral, Batticaloa -> religion.place_of_worship.religion",
      "answer_types":["religion.religion"],
    },
    {
      "start_type":["religion.place_of_worship"],
      "path":"St. Mary's Cathedral, Batticaloa -> religion.place_of_worship.religious_organization -> religion.religious_organization.religion",
      "answer_types":["religion.religion"],
    },
    {
      "start_type":["location.location"],
      "path":"St. Mary's Cathedral, Batticaloa -> religion.place_of_worship.religion",
      "answer_types":["religion.religion"],
    }
]
}

# Example 4
Question: The country with the National Anthem of Bolivia borders which nations?
Topic Entities:["National Anthem of Bolivia"]
Topic Entities Types: [["government.national_anthem"]]
Output: {
"National Anthem of Bolivia":[
    {
      "start_type":["government.national_anthem"],
      "path":"National Anthem of Bolivia -> government.national_anthem_of_a_country.country -> location.country.adjoins -> location.adjoining_relationship.adjoins",
      "answer_types":["location.country"],
    },
    {
      "start_type":["government.national_anthem"],
      "path":"National Anthem of Bolivia -> government.national_anthem.country -> location.location.adjoin_s -> location.adjoining_relationship.adjoins",
      "answer_types":["location.country","location.location"],
    },
    {
      "start_type":["government.national_anthem"],
      "path":"National Anthem of Bolivia -> government.national_anthem_of_a_country.country -> location.country.shares_border_with",
      "answer_types":["location.country"],
    }
]
}

Now begin for the real input:
"""


def build_semantic_prompt(question: str, topic_entity: dict) -> str:
    topic_names = [name for mid, name in topic_entity.items()]
    topic_json = json.dumps(topic_names, ensure_ascii=False)
    types_list = []
    lines = ["Topic Entities Types:"]
    for mid, name in topic_entity.items():
        try:
            raw = get_entity_types(mid)
        except:
            raw = []
        clean = filter_valid_types(raw)
        types_list.append(clean)
        lines.append(f"- {name}: {clean}")
    types_json = json.dumps(types_list, ensure_ascii=False)
    type_block = "\n".join(lines)
    prompt = (
            RELATION_PLAN_PROMPT
            + "\n"
            + type_block
            + "\n\n"
            + f"Question: {question}\n"
            + f"Topic Entities: {topic_json}\n"
            + f"Topic Entities Types: {types_json}\n"
            + "Output only the JSON dict.\n"
    )
    return prompt