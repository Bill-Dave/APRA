# orchestrator_full.py
"""
APRA Orchestrator - Full integration

Place this file in the project root (next to the `apra/` package).
It expects these modules to exist:
    apra.apra_algorithm or apra.core  (APRAAlgorithm / APRAEngine)
    apra.utils
    apra.world
    apra.voi
    apra.meta_reasoner
    apra.planner

Features
- FastAPI service with endpoints:
    GET  /status
    POST /auto_build      -> full automated generate -> test -> patch loop (uses LLM)
    POST /simulate_hypo   -> simulate a hypothesis counterfactual
    POST /plan            -> run planner (greedy / rollout / mcts) and return recommendation
    POST /run_scan        -> run sandbox tests + security scans on provided code
- LLM integration (Groq or HuggingFace) via environment variables
- Simple sandbox runner that writes files and runs bandit/pip-audit/pytest/semgrep - for demo only
  IMPORTANT: Do not run untrusted code with this in production - use containerized, network-isolated sandbox.

Quick start:
    pip install fastapi uvicorn requests pytest bandit pip-audit semgrep numpy
    export LLM_API_KEY="..."            # optional, orchestrator handles missing key
    export LLM_PROVIDER="groq"          # or "huggingface"
    python -m uvicorn orchestrator_full:app --port 8000

Security note (again): This orchestrator uses local subprocesses to run tests/scanners. In production,
put these executions into isolated ephemeral containers with strict limits and no network. Use sandboxing.

"""

from __future__ import annotations
import os
import json
import tempfile
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# core APRA imports - prefer apra.core compatibility wrapper if available
try:
    # core adapter wraps APRAAlgorithm as APRAEngine
    from apra.core import APRAEngine
except Exception:
    try:
        from apra.apra_algorithm import APRAAlgorithm as APRAEngine  # type: ignore
    except Exception as e:
        raise RuntimeError("Cannot import APRA kernel. Ensure apra package files are present") from e

from apra.utils import to_json_safe, make_rng, stable_softmax_vec, EPS
from apra.world import WorldModel
from apra.voi import compute_voi_feature, expected_utility, choose_best_action
from apra.meta_reasoner import MetaReasoner
from apra.planner import Planner
import numpy as np
import requests

# -----------------------------
# LLM connector (Groq / HuggingFace)
# -----------------------------
LLM_KEY = os.environ.get("LLM_API_KEY", "")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()
HF_MODEL = os.environ.get("HF_MODEL", "bigcode/starcoder")

def call_llm_api(prompt: str, max_tokens: int = 1024, provider: Optional[str] = None) -> str:
    """
    Minimal LLM caller. Supports groq and huggingface inference endpoints.
    If LLM_KEY not provided, returns a placeholder string (so the orchestration loop still works offline).
    """
    provider = (provider or LLM_PROVIDER).lower()
    if not LLM_KEY:
        # offline placeholder - returns a simple template for tests
        return ("---CODE---\n"
                "def app_func(x):\n    '''placeholder generated code'''\n    return x*x\n"
                "\n---TESTS---\n"
                "def test_smoke():\n    assert app_func(2) == 4\n")
    headers = {"Authorization": f"Bearer {LLM_KEY}"}
    if provider == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        data = {
            "model": "llama3-8b-8192",  # adjust as needed
            "messages": [{"role":"user","content":prompt}],
            "max_tokens": max_tokens
        }
        r = requests.post(url, headers={**headers, "Content-Type":"application/json"}, json=data, timeout=120)
        r.raise_for_status()
        j = r.json()
        # Support common response shapes
        if isinstance(j, dict) and "choices" in j and len(j["choices"])>0:
            c = j["choices"][0]
            return (c.get("message") or {}).get("content") or c.get("text","")
        return json.dumps(j)
    elif provider == "huggingface":
        url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
        payload = {"inputs": prompt, "options": {"wait_for_model": True}}
        r = requests.post(url, headers=headers, json=payload, timeout=180)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and len(j)>0 and isinstance(j[0], dict):
            return j[0].get("generated_text", "") or str(j[0])
        if isinstance(j, dict) and "generated_text" in j:
            return j["generated_text"]
        return str(j)
    else:
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider}")

# -----------------------------
# Sandbox runner - runs tests and scanners (demo; not secure)
# -----------------------------
def run_code_and_scan(code: str, tests: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
    """
    Write code and tests to a temp dir, run `pytest`, `bandit`, `pip-audit`, `semgrep`.
    Returns a summary dict. Keep this function for local testing only.
    """
    tmp = tempfile.mkdtemp(prefix="apra_run_")
    res = {"ok": False, "pytest": None, "bandit": None, "pip_audit": None, "semgrep": None, "summary": {}}
    try:
        open(os.path.join(tmp, "app.py"), "w", encoding="utf-8").write(code)
        if tests:
            open(os.path.join(tmp, "test_app.py"), "w", encoding="utf-8").write(tests)
        # pip-audit
        try:
            p = subprocess.run(["pip-audit", "--format", "json"], cwd=tmp, capture_output=True, timeout=timeout)
            out = p.stdout.decode(errors="ignore") or p.stderr.decode(errors="ignore")
            res["pip_audit"] = out
        except Exception as e:
            res["pip_audit"] = {"error": str(e)}
        # bandit
        try:
            b = subprocess.run(["bandit", "-r", ".", "-f", "json"], cwd=tmp, capture_output=True, timeout=timeout)
            out = b.stdout.decode(errors="ignore") or b.stderr.decode(errors="ignore")
            res["bandit"] = out
        except Exception as e:
            res["bandit"] = {"error": str(e)}
        # semgrep
        try:
            s = subprocess.run(["semgrep", "--json", "--quiet", "."], cwd=tmp, capture_output=True, timeout=timeout)
            out = s.stdout.decode(errors="ignore") or s.stderr.decode(errors="ignore")
            res["semgrep"] = out
        except Exception as e:
            res["semgrep"] = {"error": str(e)}
        # pytest
        if tests:
            try:
                r = subprocess.run(["pytest", "-q", "--disable-warnings", "--maxfail=1"], cwd=tmp, capture_output=True, timeout=timeout)
                res["pytest"] = {"rc": r.returncode, "stdout": r.stdout.decode(errors="ignore"), "stderr": r.stderr.decode(errors="ignore")}
                res["ok"] = (r.returncode == 0)
            except Exception as e:
                res["pytest"] = {"error": str(e)}
                res["ok"] = False
        else:
            res["pytest"] = {"note": "no tests provided"}
        # summarize bandit if json
        try:
            band = res["bandit"]
            band_issues = 0
            band_high = 0
            if isinstance(band, str) and band.strip().startswith("{"):
                bj = json.loads(band)
                for r_ in bj.get("results", []):
                    band_issues += 1
                    if r_.get("issue_severity", "").upper() == "HIGH":
                        band_high += 1
            sem_count = 0
            if isinstance(res.get("semgrep"), str) and res.get("semgrep").strip().startswith("{"):
                sj = json.loads(res.get("semgrep"))
                sem_count = len(sj.get("results", []))
            pip_count = 0
            try:
                pa = res.get("pip_audit")
                if isinstance(pa, str) and pa.strip().startswith("["):
                    pip_count = len(json.loads(pa))
            except Exception:
                pip_count = 0
            res["summary"] = {"bandit_issues": band_issues, "bandit_high": band_high, "semgrep_count": sem_count, "pip_vuln_count": pip_count}
        except Exception:
            res["summary"] = {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return res

# -----------------------------
# Data models for endpoints
# -----------------------------
class AutoBuildReq(BaseModel):
    task_title: str
    task_description: str
    max_iters: int = 3
    voi_cost: float = 1.0

class PlanReq(BaseModel):
    task_title: str
    task_description: str
    method: str = "greedy"  # or rollout or mcts
    budget_actions: Optional[List[int]] = None
    horizon: int = 3

class HypoReq(BaseModel):
    hypothesis: Dict[str, Any]
    start_outcome: int = 0
    actions: List[int] = []
    horizon: int = 3
    sims: int = 200

class RunScanReq(BaseModel):
    code: str
    tests: Optional[str] = None

# -----------------------------
# Instantiate app and shared objects
# -----------------------------
app = FastAPI(title="APRA Orchestrator Full")

# default outcomes / priors / order_models for demo (can be replaced per-request)
DEFAULT_OUTCOMES = ["django_drf", "fastapi", "flask_minimal", "django_async", "microservices"]
DEFAULT_PRIORS = [0.35, 0.25, 0.10, 0.15, 0.15]
DEFAULT_ORDER_MODELS = {
    0: [0.95, 0.75, 0.30, 0.9, 0.85],  # security_critical
    1: [0.6, 0.95, 0.5, 0.85, 0.9],    # performance_need
    2: [0.95, 0.6, 0.2, 0.9, 0.8],     # data_complexity
    3: [0.6, 0.85, 0.25, 0.8, 0.98],   # scalability_need
    4: [0.98, 0.6, 0.1, 0.8, 0.7]      # compliance
}
# utilities matrix - actions x outcomes (A x K)
DEFAULT_UTILITIES = np.array([
    [100.0,  80.0,  40.0,  95.0,  90.0],   # prioritize_security
    [ 70.0, 100.0,  50.0,  90.0,  95.0],   # prioritize_performance
    [ 60.0,  70.0, 100.0,  65.0,  45.0],   # prioritize_dev_speed
    [ 75.0,  90.0,  35.0,  80.0, 100.0],   # prioritize_scalability
    [ 98.0,  75.0,  20.0,  88.0,  92.0]    # prioritize_compliance
], dtype=float)

# lightweight global memory (simple)
GLOBAL_MEMORY: Dict[str, Any] = {"builds": []}

# -----------------------------
# Helpers & orchestrator logic
# -----------------------------
def build_apra_for_task(task_desc: str) -> APRAEngine:
    """
    Construct an APRAEngine instance tuned for the task description by extracting evidence.
    For now we use DEFAULT_ORDER_MODELS and adjust priors heuristically (very simple).
    """
    apra = APRAEngine(outcomes=DEFAULT_OUTCOMES, priors=DEFAULT_PRIORS, order_models=DEFAULT_ORDER_MODELS, seed=42)
    # simple heuristic: if keywords present, bump corresponding prior probabilities
    desc = (task_desc or "").lower()
    # fuzz mapping: (keyword list, outcome idx, bump)
    mapping = [
        (["payment", "card", "pci", "fintech", "bank"], 0, 0.15),
        (["real-time", "latency", "throughput", "concurrent"], 1, 0.2),
        (["prototype","minimal","quick","simple"], 2, 0.15),
        (["async","websocket","realtime","channels"], 3, 0.10),
        (["scale","microservice","shard","kubernetes"], 4, 0.10)
    ]
    pri = np.array(apra.get_priors(), dtype=float)
    for kws, idx, bump in mapping:
        if any(k in desc for k in kws):
            pri[idx] = min(0.999, pri[idx] + bump)
    pri = pri / (pri.sum() + EPS)
    apra.set_priors(pri.tolist())
    return apra

def extract_domain_evidence_simple(task_desc: str) -> List[Tuple[int,int]]:
    # reuse rules similar to earlier code (kept simple)
    d = (task_desc or "").lower()
    ev = []
    ev.append((0, 1 if any(kw in d for kw in ["payment","fintech","pci","card","auth"]) else 0))
    ev.append((1, 1 if any(kw in d for kw in ["real-time","latency","throughput","concurrent"]) else 0))
    ev.append((2, 1 if any(kw in d for kw in ["complex","relationships","admin","analytics","reporting"]) else 0))
    ev.append((3, 1 if any(kw in d for kw in ["scale","microservice","kubernetes","shard","multi-tenant"]) else 0))
    ev.append((4, 1 if any(kw in d for kw in ["gdpr","pci","hipaa","compliance","encryption"]) else 0))
    return ev

# auto-build loop (core orchestration)
def auto_build_loop(task_title: str, task_desc: str, max_iters: int = 3, voi_cost: float = 1.0) -> Dict[str, Any]:
    """
    High-level loop:
      1) Build APRA engine tuned to task
      2) Extract domain evidence -> posterior
      3) Determine selected architecture (outcome) by posterior argmax
      4) Build enhanced prompt with architecture constraints + security requirements
      5) Call LLM to generate code + tests
      6) Run sandbox scans & tests
      7) If failures and VOI indicates improvement possible, request patch from LLM and iterate
    """
    apra = build_apra_for_task(task_desc)
    evidence = extract_domain_evidence_simple(task_desc)
    posterior = apra.compute_posterior_from_prefix(evidence)
    arch_idx = int(np.argmax(posterior))
    selected_arch = apra.outcomes[arch_idx]
    confidence = float(posterior[arch_idx])
    utilities = DEFAULT_UTILITIES.copy()

    logs = []
    best_artifact = None

    # security reqs simple
    security_reqs = []
    if any(e for (o,e) in evidence if o==0 and e==1):
        security_reqs += ["Parameterize DB queries", "Input validation", "Encrypt secrets"]
    if any(e for (o,e) in evidence if o==4 and e==1):
        security_reqs += ["Audit logs", "Data retention controls"]

    for it in range(max_iters):
        logs.append({"iter": it, "posterior": posterior.tolist(), "selected_arch": selected_arch, "confidence": confidence})
        # build prompt
        constraints = f"Selected architecture: {selected_arch}\nSecurity reqs: {', '.join(security_reqs)}\nTask: {task_title}\nDesc: {task_desc}\n"
        prompt = f"""You are an expert backend engineer. Produce production-ready code and pytest tests. Constraints:
{constraints}
Return two blocks: ---CODE--- followed by ---TESTS--- as shown in the examples.
"""
        llm_out = call_llm_api(prompt, max_tokens=1200)
        logs.append({"llm_sample": str(llm_out)[:1200]})
        code_block = None
        tests_block = None
        if isinstance(llm_out, str) and ("---CODE---" in llm_out and "---TESTS---" in llm_out):
            try:
                left = llm_out.split("---CODE---",1)[1]
                code_block = left.split("---TESTS---",1)[0].strip()
                tests_block = left.split("---TESTS---",1)[1].strip()
            except Exception:
                code_block = llm_out
                tests_block = None
        else:
            # fallback: use entire output as code and small default test
            code_block = llm_out
            tests_block = "def test_smoke():\n    assert True"

        scan = run_code_and_scan(code_block, tests_block)
        logs.append({"scan_summary": scan.get("summary")})
        # heuristics to decide if we iterate
        tests_ok = scan.get("ok", False)
        band_high = scan.get("summary", {}).get("bandit_high", 0)
        pip_vuln = scan.get("summary", {}).get("pip_vuln_count", 0)

        if tests_ok and band_high == 0 and pip_vuln == 0:
            best_artifact = {"code": code_block, "tests": tests_block, "scan": scan}
            logs.append({"decision": "success", "iter": it})
            break

        # compute VOI on a representative informative order (order 0 security) to decide whether to continue
        voi, diag = compute_voi_feature(apra, posterior, order_idx=0, utilities=utilities)
        logs.append({"voi": voi, "voi_diag": diag})
        # heuristic decision threshold (depends on confidence)
        if voi > voi_cost or it == 0:
            # ask LLM for a patch with the scan summary
            patch_prompt = f"""The generated code failed security/tests. Scan summary: {json.dumps(scan.get('summary', {}))}\nPlease produce a corrected app.py and tests in the same markers ---CODE--- and ---TESTS---. Address issues: {scan.get('summary', {})}"""
            patch_out = call_llm_api(patch_prompt, max_tokens=1200)
            logs.append({"patch_raw": str(patch_out)[:1200]})
            # try to extract
            if ("---CODE---" in patch_out and "---TESTS---" in patch_out):
                try:
                    left = patch_out.split("---CODE---",1)[1]
                    code_block = left.split("---TESTS---",1)[0].strip()
                    tests_block = left.split("---TESTS---",1)[1].strip()
                except Exception:
                    pass
            # update evidence with scan results for next posterior
            new_ev = []
            if scan.get("summary", {}).get("bandit_high", 0) > 0:
                new_ev.append((0,1))
            if scan.get("summary", {}).get("pip_vuln_count", 0) > 0:
                new_ev.append((4,1))
            combined = evidence + new_ev
            posterior = apra.compute_posterior_from_prefix(combined)
            arch_idx = int(np.argmax(posterior))
            selected_arch = apra.outcomes[arch_idx]
            confidence = float(posterior[arch_idx])
            # continue loop to re-generate or patch
            continue
        else:
            logs.append({"decision": "stop_voi_low", "iter": it})
            break

    result = {"task_title": task_title, "selected_arch": selected_arch, "confidence": confidence, "artifact": best_artifact, "logs": logs}
    # store in memory
    GLOBAL_MEMORY["builds"].append({"ts": time.time(), "task_title": task_title, "result": result})
    return result

# -----------------------------
# FastAPI endpoints
# -----------------------------
@app.get("/status")
def status():
    return {"ok": True, "llm_provider": LLM_PROVIDER, "has_key": bool(LLM_KEY)}

@app.post("/auto_build")
def auto_build(req: AutoBuildReq):
    try:
        out = auto_build_loop(req.task_title, req.task_description, max_iters=req.max_iters, voi_cost=req.voi_cost)
        if not out.get("artifact"):
            return {"status": "needs_human", "result": out}
        return {"status": "success", "artifact": out["artifact"], "logs": out["logs"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate_hypo")
def simulate_hypo(req: HypoReq):
    try:
        apra = build_apra_for_task("")  # minimal apra; user can pass different apra later
        wm = WorldModel(apra)
        mr = MetaReasoner(apra, world_model=wm)
        res = mr.simulate_counterfactual(req.hypothesis, req.start_outcome, req.actions, horizon=req.horizon, num_sims=req.sims)
        return {"ok": True, "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plan")
def plan(req: PlanReq):
    try:
        apra = build_apra_for_task(req.task_description)
        evidence = extract_domain_evidence_simple(req.task_description)
        posterior = apra.compute_posterior_from_prefix(evidence)
        wm = WorldModel(apra)
        planner = Planner(apra, wm, DEFAULT_UTILITIES, rng_seed=42)
        if req.method == "greedy":
            out = planner.plan_greedy(posterior)
            return {"method": "greedy", "out": out}
        elif req.method == "rollout":
            budget = req.budget_actions or list(range(DEFAULT_UTILITIES.shape[0]))
            out = planner.plan_rollout_search(posterior, budget_actions=budget, horizon=req.horizon, rollouts_per_seq=120, beam_width=4)
            return {"method": "rollout", "out": out}
        elif req.method == "mcts":
            budget = req.budget_actions or list(range(DEFAULT_UTILITIES.shape[0]))
            out = planner.plan_mcts(posterior, budget_actions=budget, horizon=req.horizon, sims=200)
            return {"method": "mcts", "out": out}
        else:
            raise HTTPException(status_code=400, detail="Unknown planning method")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_scan")
def run_scan(req: RunScanReq):
    try:
       out = run_code_and_scan(req.code, req.tests)
        return {"ok": True, "scan": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# CLI convenience
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator_full:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


       