# Freebase Knowledge Graph Setup Guide

This guide provides step-by-step instructions for setting up a local Freebase knowledge graph using OpenLink Virtuoso.

## Prerequisites

| Component | Version | Source |
|-----------|---------|--------|
| OpenLink Virtuoso | 7.2.5 | [SourceForge Download](https://sourceforge.net/projects/virtuoso/files/virtuoso/7.2.5/virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz/download) |
| Python | 3.x | - |
| Freebase RDF Dump | - | [Google Developers](https://developers.google.com/freebase) |

## Setup Instructions

### Option A: Use Pre-processed Virtuoso Database (Recommended)

The official Freebase dump has some literal format issues that may cause import failures. The [dki-lab/Freebase-Setup](https://github.com/dki-lab/Freebase-Setup) project provides a pre-processed Virtuoso database file:

```bash
# Download pre-processed database (~53GB)
wget https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip

# Extract
unzip virtuoso_db.zip
```

Then skip to **Step 2** below.

### Option B: Import from Raw Data

#### Step 1: Data Preprocessing

The raw Freebase dump contains multilingual data. We filter it to retain only English and numeric triplets.

```bash
# Decompress the Freebase dump (~400GB uncompressed)
gunzip -c freebase-rdf-latest.gz > freebase

# Filter English triplets (~125GB after filtering)
nohup python -u FilterEnglishTriplets.py < freebase > FilterFreebase 2> log_err &
```

**Note**: The `FilterEnglishTriplets.py` script filters non-English literals. You can also use the [fix_freebase_literal_format.py](https://github.com/dki-lab/Freebase-Setup/blob/master/fix_freebase_literal_format.py) script from dki-lab to fix literal format issues. For more data processing tools, see [nchah/freebase-triples](https://github.com/nchah/freebase-triples).

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

Some entities in the Freebase dump have incomplete relationships. To supplement this data, you can import Freebase-Wikidata mappings:

1. Download the mappings from [Google Developers - Freebase](https://developers.google.com/freebase) (see "Freebase/Wikidata Mappings" section)
2. Use the same import method described in Step 4

For additional mapping tools, see [google/freebase-wikidata-converter](https://github.com/google/freebase-wikidata-converter).

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
| Literal format errors | Use [fix_freebase_literal_format.py](https://github.com/dki-lab/Freebase-Setup/blob/master/fix_freebase_literal_format.py) to preprocess data |

## References

- [Virtuoso Bulk RDF Loader Documentation](http://vos.openlinksw.com/owiki/wiki/VOS/VirtBulkRDFLoader)
- [Freebase Basic Concepts](https://developers.google.com/freebase/guide/basic_concepts)
- [Freebase-MQL Overview](https://github.com/nchah/freebase-mql)
- [dki-lab/Freebase-Setup](https://github.com/dki-lab/Freebase-Setup) - Pre-processed database and utilities
