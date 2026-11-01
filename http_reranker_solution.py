
class HTTPMemoryReranker:
    def __init__(self, endpoint, *, client, batch_size=32, policy_id="http-cross-encoder-v1"):
        self.endpoint = endpoint
        self._client = client
        self.batch_size = batch_size
        self.policy_id = policy_id
        self.supports_atomic_memory = True

    async def rerank(self, query, candidates, limit, /):
        import math
        if len(candidates) < 1 or limit < 1 or limit > len(candidates):
            raise ValueError("Invalid rerank bounds")

        batches = []
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            payload = {
                "query": query,
                "candidates": [{"rank": i + j + 1, "text": getattr(c, "text", str(c))} for j, c in enumerate(batch)],
            }
            batches.append(payload)

        all_scores = []
        try:
            for payload in batches:
                response = await self._client.post(self.endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
                if "scores" not in data: raise ValueError("Missing scores")
                for item in data["scores"]:
                    if "rank" not in item or "score" not in item: raise ValueError("Bad score")
                    if not math.isfinite(item["score"]): raise ValueError("Not finite")
                    all_scores.append(item)
        except Exception as e:
            raise ValueError(f"HTTP failed: {e}")

        all_scores.sort(key=lambda x: (-x["score"], x["rank"]))
        return [item["rank"] for item in all_scores[:limit]]
