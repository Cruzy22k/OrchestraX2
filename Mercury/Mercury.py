import asyncio
import json
import sys
from pydantic import BaseModel
from agentaus_client import AgentausClient
from datetime import datetime



client = AgentausClient()

# --- STEP 1: Define the Blueprint (Structured Output) ---
# agentaus_client has no response_format/parse, so we force JSON structure by
# making the Manager call a tool whose arguments carry the TaskPlan schema.
class SubTask(BaseModel):
    task_name: str
    role: str          # e.g., "Researcher", "Coder", "Copywriter"
    instruction: str   # The exact prompt for the worker AI

class TaskPlan(BaseModel):
    subtasks: list[SubTask]

# Mirrors the TaskPlan schema so the model returns arguments we can validate.
PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_task_plan",
        "description": "Submit the breakdown of the big goal into subtasks.",
        "parameters": {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_name": {"type": "string",
                                          "description": "Short name of the task"},
                            "role": {"type": "string",
                                     "description": "Persona, e.g. Researcher, Coder, Copywriter"},
                            "instruction": {"type": "string",
                                            "description": "Exact prompt for the worker AI"},
                        },
                        "required": ["task_name", "role", "instruction"],
                    },
                }
            },
            "required": ["subtasks"],
        },
    },
}

#The Manager AI (Splits the big task)
async def create_plan(big_goal: str) -> TaskPlan:
    print("\n================ MANAGER =================\n")
    print(f"Manager is breaking down: '{big_goal}'...\n")

    # Force a tool call with tool_choice="required" so the reply's arguments are
    # structured JSON we can validate against TaskPlan (to_thread keeps the sync
    # client out of the event loop).
    data = await asyncio.to_thread(
        client.chat,
        [
            {"role": "system",
             "content": "You are a project manager. Break the user's goal into 3 to 5 "
                        "independent, actionable subtasks for smaller AI workers. "
                        "Return the plan via the submit_task_plan tool."},
            {"role": "user", "content": big_goal},
        ],
        tools=[PLAN_TOOL],
        tool_choice="required",
    )

    calls = client.tool_calls(data)
    if calls and calls[0].get("arguments"):
        return TaskPlan.model_validate(calls[0]["arguments"])
    raise RuntimeError(
        "Manager did not return a structured plan via submit_task_plan. Response: "
        + str(data)[:300]
    )

# Load the boundaries every worker must follow (editable in Boundries.JSON).
def load_boundaries(path: str = "Boundries.JSON") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            rules = json.load(f).get("rules", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not load boundaries from {path}: {e}")
        return ""
    if not rules:
        return ""
    return "Global boundaries all workers must follow:\n" + "\n".join(
        f"- {rule}" for rule in rules
    )


#The Worker AI (Executes one small task)
async def run_worker(task: SubTask) -> str:
    print("\n================ WORKER =================\n")
    print(f"Worker [{task.role}] started: {task.task_name}")

    system_content = f"You are an expert {task.role}. Complete your assigned task concisely."
    boundaries = load_boundaries()
    if boundaries:
        system_content += "\n\n" + boundaries

    content = await asyncio.to_thread(
        client.reply_text,
        [
            {"role": "system", "content": system_content},
            {"role": "user", "content": task.instruction},
        ],
    )
    print(f"Worker [{task.role}] finished: {task.task_name}")
    return f"### {task.task_name} ({task.role})\n{content}"

#The Aggregator (Combines everything)
async def synthesize_results(big_goal: str, worker_outputs: list[str]) -> str:
    print("\n================ SYNTHESIZER =================\n")
    print("\nManager is combining all worker results...")
    combined_work = "\n\n".join(worker_outputs)

    return await asyncio.to_thread(
        client.reply_text,
        [
            {"role": "system",
             "content": "Combine the following completed subtasks into one cohesive, polished final response."},
            {"role": "user", "content": f"Original Goal: {big_goal}\n\nWorker Results:\n{combined_work}"},
        ],
    )


def get_big_task(argv) -> str:
    """Read a task from a file argument, or multi-line terminal paste.

    Priority:
      1. argv[1] treated as a file path (best for large pastes) — read whole file.
      2. Otherwise read multi-line input from the terminal until the user
         types 'END' alone on a line or sends EOF (Ctrl-D / Ctrl-Z).
    """
    import sys

    # 1) File input: python Divide_Et_Impera.py mytask.txt
    if len(argv) > 1:
        path = argv[1]
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except OSError as e:
            print(f"Could not read file '{path}': {e}", file=sys.stderr)
            raise SystemExit(1)
        if not content:
            print(f"File '{path}' is empty.", file=sys.stderr)
            raise SystemExit(1)
        return content

    # 2) Multi-line terminal paste. Empty 'END' line or EOF ends input.
    print("Enter task (paste it; finish with a line containing only 'END', or Ctrl-D):")
    lines = []
    for line in sys.stdin:
        if line.strip() == "END":
            break
        lines.append(line.rstrip("\n"))
    content = "\n".join(lines).strip()
    if not content:
        print("No task entered.", file=sys.stderr)
        raise SystemExit(1)
    return content


async def main():

    big_task = get_big_task(sys.argv)
    # For example:"Create a launch plan for a new eco-friendly coffee brand, including brand names, target audience, and a 3-post social media teaser."

    #Split the task
    plan = await create_plan(big_task)

    #Run all smaller AIs in parallel (at the exact same time)
    worker_coroutines = [run_worker(task) for task in plan.subtasks]
    results = await asyncio.gather(*worker_coroutines)

    #Combine into the final answer
    final_output = await synthesize_results(big_task, results)

    #print("\n================ FINAL OUTPUT ================\n")
    #print(final_output)


    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"output_{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(final_output)
    print(f"\nFinal output written to {filename}")

if __name__ == "__main__":
    asyncio.run(main())
