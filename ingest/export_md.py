import os, json

EXPORT_DIR = "/docs"
os.makedirs(EXPORT_DIR, exist_ok=True)

for fname in os.listdir("/data"):
    if fname.endswith(".json"):
        with open(f"/data/{fname}") as f:
            data = json.load(f)
        summary = f"# Documentation for {fname}\n\n"
        summary += f"## Keys\n\n" + "\n".join(f"- **{k}**: {type(v).__name__}" for k, v in data.items())
        with open(os.path.join(EXPORT_DIR, fname.replace(".json", ".md")), "w") as out:
            out.write(summary)