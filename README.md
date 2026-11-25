# OrchestraX2
OrchestraX2 is a triaging Ai agentic chatbot that creates plans and synthesizes infomation from a band of employed Ai agents

Logical foundation :

1. The manager
- has create plan method, breaks prompt up into smaller prompts
- by passing BaseModel, the AI is blocked from outputting any character that violates our JSON scheme
- it must output a predictable array of objects (task_name, role, instruction) that your Python code can handle

2. The workers
- Isolation: Each worker gets a blank slate (messages=[...]) containing only its specific persona and single instruction. No distractions.
- Concurrency (asyncio.gather): Instead of running tasks sequentially (Task 1 -> wait -> Task 2 -> wait), asyncio opens multiple network connections simultaneously. Your Python script fires all 5 prompts at once, pauses until all 5 return, and collects them into a single list.

3. The Aggregator (synthesize_results) — State Reduction
- The Core Logic: Now you have a Python list of 5 disconnected text blocks (["output 1", "output 2", ...]). They might repeat information, use different tones, or lack transitions.
- How it works: You concatenate (join) those isolated outputs into a single master context and pass them back to a smart model alongside the original user prompt. Because the AI isn't doing the heavy lifting from scratch anymore—it is only editing and arranging pre-written material—it can easily produce a massive, cohesive final output without losing track of the big picture