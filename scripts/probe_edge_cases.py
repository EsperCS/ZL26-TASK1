import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "train")
from zaic_tools import ToolEnv

env = ToolEnv("train", "train_0001")

edge_cases = [
    ("Uninstalled tool validation", "door_lock.set_lock", {"lock_id": "l_01", "locked": True}),
    ("Non-existent catalog tool validation", "fake_app.do_magic", {}),
    ("Missing required parameter", "smart_home.light_set", {"power": "on"}),
    ("Empty string as required parameter", "smart_home.light_set", {"device_id": "", "power": "on"}),
    ("Empty list as required parameter", "sys.ask_missing_info", {"tool_name": "sms.send", "params": []}),
    ("x-one-of both missing", "sms.send", {"content": "hello"}),
    ("x-one-of valid contact_id", "sms.send", {"contact_id": "c_08", "content": "hello"}),
    ("x-one-of valid phone_number", "sms.send", {"phone_number": "0912345678", "content": "hello"}),
    ("Enum violation", "smart_home.light_set", {"device_id": "d1", "power": "dimmed"}),
    ("Date format invalid regex (slash)", "package_tracking.schedule_pickup", {"address": "123", "date": "2026/11/20"}),
    ("Date format invalid calendar day (Feb 30)", "package_tracking.schedule_pickup", {"address": "123", "date": "2026-02-30"}),
    ("Datetime format missing 'T'", "tasks.add", {"title": "Test", "due": "2026-11-29 09:00:00"}),
    ("Datetime format valid ISO with 'T'", "tasks.add", {"title": "Test", "due": "2026-11-29T09:00:00+07:00"}),
    ("Call invalid schema -> rejection and calls list", "smart_home.light_set", {"power": "off"}),
    ("Call non-existent entity ID", "calendar.update_event", {"event_id": "non_existent_evt", "start": "2026-11-29T10:00:00+07:00"}),
]

print("--- EXECUTING EDGE CASES ---")
for name, tool_name, args in edge_cases:
    if "Call" in name:
        res = env.call(tool_name, args)
        print(f"[{name}] {tool_name}({args}) -> Call Result: {res}")
    else:
        res = env.validate(tool_name, args)
        print(f"[{name}] {tool_name}({args}) -> Validate Result: {res}")

print(f"\nTotal calls in env.get_tool_calls(): {len(env.get_tool_calls())}")
for c in env.get_tool_calls():
    print("  Recorded call:", c)
