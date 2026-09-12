# octoctoctopus - Job Applier 

A personal automation tool that fills out applications on your behalf. User is expected to submit job posting links to the agent. The agent drafts form answers and a cover letter using a local LLM.
Answers for open-ended questions are synthesized from the memory of past answers. If no previous responses accurately answer a given question - the agent pauses and asks the user to manually fill out the information.
Nothing is submitted without the user's explicit approval. 

**Current Tech Stack**: Python, SQLite3, Ollama + Qwen (for local inference), bge-small (embedding model for memory), Playwright (for form fill automation)

Note: None of the model/inference choices above are hard requirements. They are what fits **my** particular rig (8GB of VRAM). It would suffice to modify or add a new class implementing the relevant protocol interface (LLMClient or EmbeddingClient), leaving the rest of the logic untouched. 
