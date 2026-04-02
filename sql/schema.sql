-- schema.sql - Database schema for RAG recommendation system

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS businesses (
    id                UUID        PRIMARY KEY,
    owner_id          UUID        NOT NULL,
    verified          BOOL        NOT NULL DEFAULT FALSE,
    reputation        FLOAT       NOT NULL CHECK (reputation >= 0.0 AND reputation <= 5.0),
    name              TEXT        NOT NULL,
    industry          TEXT        NOT NULL,
    sub_industry      TEXT        NOT NULL,
    roles             TEXT[]      NOT NULL DEFAULT '{}',
    category          TEXT        NOT NULL,
    trade_type        TEXT        NOT NULL,
    location          TEXT        NOT NULL,
    trade_regions     TEXT[]      NOT NULL DEFAULT '{}',
    capacity          TEXT,
    certificates      TEXT[]      NOT NULL DEFAULT '{}',
    tags              TEXT[]      NOT NULL DEFAULT '{}',
    description       TEXT        NOT NULL,
    partner_goals     TEXT        NOT NULL,
    profile_embedding VECTOR(4096) NOT NULL
    search_text       TSVECTOR     GENERATED ALWAYS AS (
                        to_tsvector('english', coalesce(name, '') || ' ' || coalesce(industry, '') || ' ' || coalesce(sub_industry, '') || ' ' ||
                                    array_to_string(roles, ' ') || ' ' || coalesce(category, '') || ' ' || coalesce(trade_type, '') || ' ' ||
                                    coalesce(location, '') || ' ' || array_to_string(trade_regions, ' ') || ' ' ||
                                    coalesce(capacity, '') || ' ' || array_to_string(certificates, ' ') || ' ' ||
                                    array_to_string(tags, ' ') || ' ' || coalesce(description, '') || ' ' ||
                                    coalesce(partner_goals, '')
                        )
                    ) STORED
);

-- ---------------------------------------------------------------------------
-- GIN indexes (array containment / overlap queries)
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_businesses_roles
    ON businesses USING gin (roles);

CREATE INDEX IF NOT EXISTS idx_businesses_tags
    ON businesses USING gin (tags);

CREATE INDEX IF NOT EXISTS idx_businesses_trade_regions
    ON businesses USING gin (trade_regions);

CREATE INDEX IF NOT EXISTS idx_businesses_certificates
    ON businesses USING gin (certificates);

CREATE INDEX IF NOT EXISTS idx_businesses_search_text
    ON businesses USING gin (search_text);

-- ---------------------------------------------------------------------------
-- B-tree indexes (equality / range filters)
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_businesses_industry
    ON businesses (industry);

CREATE INDEX IF NOT EXISTS idx_businesses_category
    ON businesses (category);

CREATE INDEX IF NOT EXISTS idx_businesses_trade_type
    ON businesses (trade_type);

CREATE INDEX IF NOT EXISTS idx_businesses_location
    ON businesses (location);
