OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "qwen3-embedding"
LLM_MODEL = "llama3.1:8b"

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "rag_recommendation",
    "user": "raguser",
    "password": "password",
}


INSERT_SQL = """
INSERT INTO businesses (
    id, owner_id, verified, reputation,
    name, industry, sub_industry, roles,
    category, trade_type, location, trade_regions,
    capacity, certificates, tags,
    description, partner_goals, profile_embedding
) VALUES (
    %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s
)
ON CONFLICT (id) DO NOTHING;
"""
