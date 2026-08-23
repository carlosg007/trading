import glob, json, os

files = sorted(glob.glob("/mnt/backtest/artifacts/pipeline/**/gate_audit_*.json", recursive=True))

print("\n" + "=" * 125)
print(f"  {'STRATEGY':<33} {'SYM':<5} {'TF':<5} {'GATE 1':<12} {'GATE 2':<14} {'GATE 3':<14} {'STATUS':<14} {'EXCLUDED'}")
print("=" * 125)

passed = []

for f in files:
    try:
        strat = f.split("/artifacts/pipeline/")[1].split("/")[0]
        with open(f) as fp:
            d = json.load(fp)
            
        sym = d.get("symbol") or os.path.basename(f).replace("gate_audit_", "").replace(".json", "")
        tf = d.get("timeframe") or d.get("tf") or "N/A"
        
        # Check overall certified flag
        certified = d.get("certified", False)
        status_str = d.get("final_status") or ("CERTIFIED" if certified else "NOT CERTIFIED")
        
        # Check gates
        def get_verdict(g_name, g_num):
            # Check top level
            if f"{g_name}_verdict" in d: return str(d[f"{g_name}_verdict"]).upper()
            if f"gate{g_num}_verdict" in d: return str(d[f"gate{g_num}_verdict"]).upper()
            if g_name in d and isinstance(d[g_name], dict):
                v = d[g_name].get("verdict") or d[g_name].get("status") or d[g_name].get("passed")
                if isinstance(v, bool): return "PASS" if v else "FAIL"
                if v: return str(v).upper()
            if f"gate_{g_num}" in d and isinstance(d[f"gate_{g_num}"], dict):
                v = d[f"gate_{g_num}"].get("verdict") or d[f"gate_{g_num}"].get("status")
                if v: return str(v).upper()
            # fallback from audit string
            sel = str(d.get("scan_selection", ""))
            if "NO COMBINATION CLEARED GATE 1" in sel and g_num == 1:
                return "FAIL"
            return "FAIL" if not certified else "PASS"

        g1 = get_verdict("gate1", 1)
        g2 = get_verdict("gate2", 2)
        g3 = get_verdict("gate3", 3)
        
        excl = str(d.get("exclude_days") or d.get("excluded_days") or "-")
        
        if "CERTIFIED" in status_str and "NOT" not in status_str:
            passed.append((strat, sym, tf))
            
        print(f"  {strat:<33} {sym:<5} {tf:<5} {g1:<12} {g2:<14} {g3:<14} {status_str:<14} {excl}")
    except Exception as e:
        pass

print("=" * 125)
print(f"TOTAL CERTIFIED CONFIGURATIONS: {len(passed)}")
for p in passed:
    print(f"  🏆 {p[0]} -> {p[1]} ({p[2]})")
print("=" * 125 + "\n")
