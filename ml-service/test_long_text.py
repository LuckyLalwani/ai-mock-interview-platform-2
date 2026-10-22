import sys
from text_inference import text_predictor

paragraph = (
    "In my previous software engineering role, I was tasked with architecting a real-time distributed pipeline. "
    "I felt very confident and excited about solving the low-latency challenges because our team needed to process "
    "millions of events per second with high availability. We systematically broke down the bottlenecks, benchmarked "
    "each component, and implemented asynchronous non-blocking queues. Whenever unexpected system failures occurred, "
    "I remained calm and focused on root-cause analysis, collaborating closely with cross-functional teams to resolve them. "
)
long_text = paragraph * 20
word_count = len(long_text.split())
print(f"Total words in simulated answer: {word_count}")

result = text_predictor.predict(long_text)
print("Success:", result["success"])
print("Top Emotion:", result["emotion"])
print(f"Confidence: {result['confidence']*100:.2f}%")
print(f"Total Tokens Analyzed: {result['tokens_analyzed']}")
print(f"Chunks Analyzed: {result['chunks_analyzed']}")
print("Predictions:", result["predictions"])

assert result["success"] is True
assert result["tokens_analyzed"] > 1000, "Should analyze full token stream"
assert result["chunks_analyzed"] >= 3, "1000+ words should be broken into 3+ chunks"
assert abs(sum(result["predictions"].values()) - 1.0) < 1e-4, "Probabilities must sum to 1.0"
print("\n[PASS] 1000+ words answer completely analyzed across all sliding chunks without truncation!")
