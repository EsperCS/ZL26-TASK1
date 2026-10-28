import json

with open("train/train_tools_catalog.json", encoding="utf-8") as f:
    train_cat = json.load(f)
with open("public_test/public_test_tools_catalog.json", encoding="utf-8") as f:
    test_cat = json.load(f)

tools_dict = {}

for cat, split_name in [(train_cat, "train"), (test_cat, "public_test")]:
    for app in cat.get("apps", []):
        app_id = app["id"]
        group = app.get("group", "general")
        name_vi = app.get("name_vi", app_id)
        for tool in app.get("tools", []):
            t_name = tool["name"]
            if t_name not in tools_dict:
                tools_dict[t_name] = {
                    "app": app_id,
                    "group": group,
                    "name_vi": name_vi,
                    "kind": tool.get("kind"),
                    "description": tool.get("description", ""),
                    "params": tool.get("parameters", {}),
                    "returns": tool.get("returns", {}),
                    "split": [split_name]
                }
            else:
                tools_dict[t_name]["split"].append(split_name)

print(f"Total collected tools: {len(tools_dict)}")
groups = {t["group"] for t in tools_dict.values()}
print(f"Groups: {sorted(groups)}")
for g in sorted(groups):
    g_tools = [k for k, v in tools_dict.items() if v["group"] == g]
    print(f"Group '{g}': {len(g_tools)} tools")
