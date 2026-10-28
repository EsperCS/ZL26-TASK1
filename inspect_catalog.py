import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("train/train_tools_catalog.json", encoding="utf-8") as f:
    cat = json.load(f)

print("--- X-MATCH PARAMETERS ---")
x_match_params = {}
for app in cat["apps"]:
    for tool in app.get("tools", []):
        props = tool.get("parameters", {}).get("properties", {})
        for p_name, p_spec in props.items():
            if "x-match" in p_spec:
                t_name = tool["name"]
                x_match_params[f"{t_name}.{p_name}"] = p_spec["x-match"]
                print(f"{t_name}.{p_name}: {p_spec['x-match']}")

print(f"\nTotal x-match params: {len(x_match_params)}")
print("Distinct param names with x-match:", sorted({k.split(".")[1] for k in x_match_params}))

print("\n--- X-ONE-OF TOOLS ---")
for app in cat["apps"]:
    for tool in app.get("tools", []):
        params = tool.get("parameters", {})
        if "x-one-of" in params:
            print(f"{tool['name']}: {params['x-one-of']}")

print("\n--- X-REF PARAMETERS ---")
x_ref_params = []
for app in cat["apps"]:
    for tool in app.get("tools", []):
        props = tool.get("parameters", {}).get("properties", {})
        for p_name, p_spec in props.items():
            if "x-ref" in p_spec:
                x_ref_params.append((tool["name"], p_name, p_spec["x-ref"]))
print(f"Total x-ref params: {len(x_ref_params)}")
for t, p, ref in x_ref_params[:15]:
    print(f"{t}.{p} -> {ref}")

print("\n--- ENUMS ---")
enum_params = []
for app in cat["apps"]:
    for tool in app.get("tools", []):
        props = tool.get("parameters", {}).get("properties", {})
        for p_name, p_spec in props.items():
            if "enum" in p_spec:
                enum_params.append((tool["name"], p_name, p_spec["enum"]))
print(f"Total enum params: {len(enum_params)}")
for t, p, enum_vals in enum_params[:10]:
    print(f"{t}.{p} -> {enum_vals}")

print("\n--- FORMATS ---")
format_params = []
for app in cat["apps"]:
    for tool in app.get("tools", []):
        props = tool.get("parameters", {}).get("properties", {})
        for p_name, p_spec in props.items():
            if "format" in p_spec:
                format_params.append((tool["name"], p_name, p_spec["format"]))
print(f"Total format params: {len(format_params)}")
for t, p, fmt in format_params[:10]:
    print(f"{t}.{p} -> {fmt}")
