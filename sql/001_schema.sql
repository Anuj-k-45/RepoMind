-- PostgreSQL retrieval contract. Additive migration for existing local RepoMind databases.
CREATE TABLE IF NOT EXISTS repositories (
  id BIGSERIAL PRIMARY KEY, full_name TEXT NOT NULL UNIQUE,
  indexed_commit_sha TEXT, index_generation TEXT, indexed_at TIMESTAMPTZ
);
ALTER TABLE repositories ADD COLUMN IF NOT EXISTS indexed_commit_sha TEXT;
ALTER TABLE repositories ADD COLUMN IF NOT EXISTS index_generation TEXT;
ALTER TABLE repositories ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMPTZ;
ALTER TABLE repositories ALTER COLUMN indexed_commit_sha DROP NOT NULL;
ALTER TABLE repositories ALTER COLUMN index_generation DROP NOT NULL;
ALTER TABLE repositories ALTER COLUMN indexed_at DROP NOT NULL;

CREATE TABLE IF NOT EXISTS ingestion_runs (
  index_generation TEXT PRIMARY KEY,
  repository_id BIGINT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
  commit_sha TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('building','published','failed')),
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(), completed_at TIMESTAMPTZ,
  details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS symbols (
  id TEXT PRIMARY KEY, repository_id BIGINT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
  qualified_name TEXT NOT NULL, file_path TEXT NOT NULL, kind TEXT NOT NULL,
  start_line INT, end_line INT, commit_sha TEXT NOT NULL, index_generation TEXT NOT NULL
);
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS name TEXT;
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS signature TEXT;
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS language TEXT NOT NULL DEFAULT 'java';
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS is_test BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS source_code TEXT;
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS modifiers JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
CREATE INDEX IF NOT EXISTS idx_symbols_repo_generation ON symbols(repository_id,index_generation);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(repository_id,lower(qualified_name));
CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(repository_id,file_path,start_line);

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, repository_id BIGINT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
  file_path TEXT NOT NULL, start_line INT NOT NULL, end_line INT NOT NULL,
  content TEXT NOT NULL, commit_sha TEXT NOT NULL, index_generation TEXT NOT NULL,
  search_vector TSVECTOR
);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS symbol_id TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS symbol_name TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS language TEXT NOT NULL DEFAULT 'java';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_type TEXT NOT NULL DEFAULT 'source';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_index INT NOT NULL DEFAULT 0;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
CREATE OR REPLACE FUNCTION repomind_chunks_search_vector_update() RETURNS trigger AS $$
BEGIN
  NEW.search_vector := to_tsvector('simple', coalesce(NEW.file_path, '') || ' ' ||
    coalesce(NEW.symbol_name, '') || ' ' || coalesce(NEW.content, ''));
  RETURN NEW;
END
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS repomind_chunks_search_vector_trigger ON chunks;
CREATE TRIGGER repomind_chunks_search_vector_trigger
  BEFORE INSERT OR UPDATE OF content, file_path, symbol_name ON chunks
  FOR EACH ROW EXECUTE FUNCTION repomind_chunks_search_vector_update();
UPDATE chunks SET content=content WHERE search_vector IS NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_repo_generation ON chunks(repository_id,commit_sha,index_generation);
CREATE INDEX IF NOT EXISTS idx_chunks_search_vector ON chunks USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(repository_id,file_path,start_line);

CREATE TABLE IF NOT EXISTS repository_files (
  repository_id BIGINT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
  index_generation TEXT NOT NULL, commit_sha TEXT NOT NULL, path TEXT NOT NULL,
  category TEXT NOT NULL, language TEXT NOT NULL, size_bytes BIGINT NOT NULL,
  sha256 TEXT NOT NULL, line_count INT NOT NULL,
  PRIMARY KEY(repository_id,index_generation,path)
);
CREATE INDEX IF NOT EXISTS idx_repository_files_generation ON repository_files(repository_id,index_generation);
