create table documents (
  id bigserial primary key,
  context text,
  embedding vector (1536)
)