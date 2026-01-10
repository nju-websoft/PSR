# Freebase Knowledge Graph Setup Guide

This guide provides step-by-step instructions for setting up a local Freebase knowledge graph using OpenLink Virtuoso.

## Prerequisites

| Component | Version | Source |
|-----------|---------|--------|
| OpenLink Virtuoso | 7.2.5 | [Download](public_link) |
| Python | 3.x | - |
| Freebase RDF Dump | - | [Download](public_link) |

## Setup Instructions

### Step 1: Data Preprocessing

The raw Freebase dump contains multilingual data. We filter it to retain only English and numeric triplets using a preprocessing script.

```bash
# Decompress the Freebase dump (~400GB uncompressed)
gunzip -c freebase-rdf-latest.gz > freebase

# Filter English triplets (~125GB after filtering)
nohup python -u FilterEnglishTriplets.py < freebase > FilterFreebase 2> log_err &
```

The filtering script is available [here](public_link).

### Step 2: Install and Configure Virtuoso

```bash
# Extract Virtuoso
tar xvpfz virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz

# Initialize configuration
cd virtuoso-opensource/database/
mv virtuoso.ini.sample virtuoso.ini
```

### Step 3: Start Virtuoso Service

```bash
# Option 1: Run in foreground (for debugging)
../bin/virtuoso-t -df

# Option 2: Run in background (recommended for production)
../bin/virtuoso-t
```

### Step 4: Import Data

Connect to the database and load the filtered Freebase data:

```bash
# Connect to Virtuoso (default credentials: dba/dba)
../bin/isql 1111 dba dba
```

Execute the following SQL commands:

```sql
-- Register the data file for loading
ld_dir('.', 'FilterFreebase', 'http://freebase.com');

-- Start the import process (this may take several hours)
rdf_loader_run();
```

### Step 5: Wikidata Mapping (Optional)

Some entities in the Freebase dump have incomplete relationships. To supplement this data, you can import Wikidata mappings:

1. Download the Wikidata RDF mappings from [here](public_link)
2. Use the same import method described in Step 4

## Verification

Use the following Python script to verify your installation:

```python
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://localhost:8890/sparql"

def test_connection():
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    
    query = """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?name3
        WHERE {
            ns:m.0k2kfpc ns:award.award_nominated_work.award_nominations ?e1.
            ?e1 ns:award.award_nomination.award_nominee ns:m.02pbp9.
            ns:m.02pbp9 ns:people.person.spouse_s ?e2.
            ?e2 ns:people.marriage.spouse ?e3.
            ?e2 ns:people.marriage.from ?e4.
            ?e3 ns:type.object.name ?name3
            MINUS { ?e2 ns:type.object.name ?name2 }
        }
    """
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        results = sparql.query().convert()
        print("Connection successful!")
        print(results)
    except Exception as e:
        print(f"Database connection failed: {e}")

if __name__ == "__main__":
    test_connection()
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Ensure Virtuoso service is running |
| Import hangs | Check disk space and increase `NumberOfBuffers` in `virtuoso.ini` |
| Query timeout | Adjust `MaxQueryExecutionTime` in Virtuoso configuration |
