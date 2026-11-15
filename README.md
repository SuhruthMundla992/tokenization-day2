# Tokenization and Embeddings: The Typo Tax

Day 2 of learning how AI works: an experiment that tokenizes correctly spelled words and their typos with tiktoken (`o200k_base`) to count how many extra tokens a typo costs, then embeds each pair with OpenAI's `text-embedding-3-small` and compares them with cosine similarity (against an unrelated "banana" baseline) to check whether the meaning survives the typo.

Run it with `pip install tiktoken openai numpy python-dotenv`, copy `.env.example` to `.env` and add your OpenAI API key, then run `python day2.py`.
