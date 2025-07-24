import pytest
from src.embedding.embedder import (
    SentenceTransformerEmbedder,
    EmbedderFactory,
    cosine_similarity
)


@pytest.fixture
def sentence_embedder():
    return SentenceTransformerEmbedder()


def test_sentence_transformer_embed_texts(sentence_embedder):
    texts = ["Hello world", "How are you?", "Machine learning is great"]
    embeddings = sentence_embedder.embed_texts(texts)
    
    assert len(embeddings) == 3
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) == sentence_embedder.dimension for emb in embeddings)
    assert all(isinstance(val, float) for emb in embeddings for val in emb)


def test_sentence_transformer_embed_query(sentence_embedder):
    query = "What is machine learning?"
    embedding = sentence_embedder.embed_query(query)
    
    assert isinstance(embedding, list)
    assert len(embedding) == sentence_embedder.dimension
    assert all(isinstance(val, float) for val in embedding)


def test_embedder_factory():
    # Test creating sentence transformer embedder
    embedder = EmbedderFactory.create_embedder("sentence-transformer")
    assert isinstance(embedder, SentenceTransformerEmbedder)
    
    # Test invalid embedder type
    with pytest.raises(ValueError, match="Unknown embedder type"):
        EmbedderFactory.create_embedder("invalid-type")


def test_cosine_similarity():
    # Test identical vectors
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    assert cosine_similarity(vec1, vec2) == pytest.approx(1.0)
    
    # Test orthogonal vectors
    vec1 = [1.0, 0.0]
    vec2 = [0.0, 1.0]
    assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)
    
    # Test opposite vectors
    vec1 = [1.0, 0.0]
    vec2 = [-1.0, 0.0]
    assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)
    
    # Test zero vectors
    vec1 = [0.0, 0.0]
    vec2 = [1.0, 1.0]
    assert cosine_similarity(vec1, vec2) == 0.0


def test_embedding_similarity(sentence_embedder):
    # Test that similar texts have higher cosine similarity
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "AI and machine learning are related fields",
        "The weather is nice today"
    ]
    
    embeddings = sentence_embedder.embed_texts(texts)
    
    # Similarity between first two (related) should be higher than with third
    sim_related = cosine_similarity(embeddings[0], embeddings[1])
    sim_unrelated1 = cosine_similarity(embeddings[0], embeddings[2])
    sim_unrelated2 = cosine_similarity(embeddings[1], embeddings[2])
    
    assert sim_related > sim_unrelated1
    assert sim_related > sim_unrelated2