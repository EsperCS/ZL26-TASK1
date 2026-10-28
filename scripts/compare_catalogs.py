import json

with open("train/train_tools_catalog.json", encoding="utf-8") as f:
    train_cat = json.load(f)
with open("public_test/public_test_tools_catalog.json", encoding="utf-8") as f:
    test_cat = json.load(f)

train_apps = {a["id"]: {t["name"]: t for t in a.get("tools", [])} for a in train_cat.get("apps", [])}
test_apps = {a["id"]: {t["name"]: t for t in a.get("tools", [])} for a in test_cat.get("apps", [])}

all_app_ids = set(train_apps.keys()) | set(test_apps.keys())
common_apps = set(train_apps.keys()) & set(test_apps.keys())

print(f"Total unique apps across both: {len(all_app_ids)}")
print(f"Train apps: {len(train_apps)} (tools: {sum(len(t) for t in train_apps.values())})")
print(f"Public test apps: {len(test_apps)} (tools: {sum(len(t) for t in test_apps.values())})")
print(f"Common apps: {len(common_apps)}")
print(f"Train-only apps ({len(train_apps) - len(common_apps)}): {sorted(set(train_apps.keys()) - set(test_apps.keys()))}")
print(f"Public_test-only apps ({len(test_apps) - len(common_apps)}): {sorted(set(test_apps.keys()) - set(train_apps.keys()))}")

# Total unique tools
all_tools = set()
for a in train_apps.values():
    all_tools.update(a.keys())
for a in test_apps.values():
    all_tools.update(a.keys())
print(f"Total unique tools across both catalogs: {len(all_tools)}")

print("\n--- Public Test Only Tools & Apps ---")
for app_id in sorted(set(test_apps.keys()) - set(train_apps.keys())):
    tools = test_apps[app_id]
    print(f"App: {app_id} ({len(tools)} tools)")
    for t_name, t_spec in tools.items():
        kind = t_spec.get("kind")
        params = list((t_spec.get("parameters", {}).get("properties") or {}).keys())
        req = t_spec.get("parameters", {}).get("required", [])
        print(f"  - {t_name} [{kind}]: params={params}, required={req}")

# Check fuzzy parameters in public_test
fuzzy_test = []
for app in test_cat.get("apps", []):
    for tool in app.get("tools", []):
        props = (tool.get("parameters", {}) or {}).get("properties") or {}
        for p_name, p_spec in props.items():
            if p_spec.get("x-match") == "fuzzy":
                fuzzy_test.append(f"{tool['name']}.{p_name}")
print(f"\nTotal fuzzy params in public_test: {len(fuzzy_test)}")
print("Fuzzy params in public_test:", fuzzy_test)

# Check x-one-of in public_test
x_one_of_test = []
for app in test_cat.get("apps", []):
    for tool in app.get("tools", []):
        x1 = (tool.get("parameters", {}) or {}).get("x-one-of")
        if x1:
            x_one_of_test.append((tool["name"], x1))
print(f"\nx-one-of in public_test: {len(x_one_of_test)}")
for t, x1 in x_one_of_test:
    print(f"  {t}: {x1}")
