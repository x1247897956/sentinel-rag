"""语料抓取：CVE List v5 / GHSA(OSV) / MITRE ATT&CK / OWASP Cheat Sheets。

全部为公开源，只做只读拉取，遵守各来源自身许可（见 README「数据与合规」）。
产物：data/corpus/<source>.jsonl（每行一个 Doc 记录），另外写出 data/corpus/manifest.json。
"""

from __future__ import annotations

import json
import re
import subprocess
import tarfile
import time
from pathlib import Path

from src.config import CORPUS_DIR, CORPUS_LIMITS, RAW_DIR
from src.ingest.schema import Doc, write_jsonl

CVE_REPO = "CVEProject/cvelistV5"
ADVISORY_TARBALL = RAW_DIR / "advisory.tar.gz"
ATTACK_TARBALL = RAW_DIR / "attack.tar.gz"
OWASP_TARBALL = RAW_DIR / "owasp.tar.gz"


def _gh_api(path: str) -> object:
    """通过 gh CLI 调用 GitHub API（已认证，5000 req/h）。

    未安装 gh 或未登录时，抓取会失败并明确报错——不做静默降级（否则语料构成无法复现）。
    """
    out = subprocess.run(
        ["gh", "api", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


def _clean(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- CVE List v5


def _cve_year_dirs(year: int, want_dirs: int) -> list[str]:
    """取该年最新的 want_dirs 个编号段目录（CVE 编号越大越新）。"""
    entries = _gh_api(f"repos/{CVE_REPO}/contents/cves/{year}")
    names = sorted(
        (e["name"] for e in entries if e["type"] == "dir" and re.fullmatch(r"\d+xxx", e["name"])),
        key=lambda n: int(n[:-3]),
    )
    return names[-want_dirs:]


def _cve_files(year: int, directory: str) -> list[str]:
    entries = _gh_api(f"repos/{CVE_REPO}/contents/cves/{year}/{directory}")
    return sorted(e["name"] for e in entries if e["name"].startswith("CVE-") and e["name"].endswith(".json"))


def _fetch_cve_json(path: str) -> dict | None:
    url = f"https://raw.githubusercontent.com/{CVE_REPO}/main/{path}"
    out = subprocess.run(
        ["curl", "-sL", "--max-time", "60", url], capture_output=True, text=True, check=False
    ).stdout
    if not out.strip():
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None

def _cvss_of(metrics: list[dict]) -> tuple[str | None, float | None]:
    """返回 (severity, baseScore)，按 v3.1 → v3.0 → v4.0 → v2 的顺序取第一个可用值。"""
    order = ["cvssV3_1", "cvssV3_0", "cvssV4_0", "cvssV2_0"]
    by_key: dict[str, dict] = {}
    for m in metrics or []:
        for k in order:
            if k in m:
                by_key.setdefault(k, m[k])
    for k in order:
        v = by_key.get(k)
        if not v:
            continue
        data = v.get("cvssData", {})
        sev = v.get("baseSeverity") or data.get("baseSeverity")
        score = data.get("baseScore")
        if score is not None:
            return (sev.lower() if isinstance(sev, str) else None), float(score)
    return None, None


def _version_facts(cna: dict) -> tuple[str, list[str]]:
    """从 affected[].versions[] 提取版本区间描述；版本区间不切散。"""
    lines: list[str] = []
    products: list[str] = []
    for aff in cna.get("affected") or []:
        vendor = aff.get("vendor") or ""
        product = aff.get("product") or ""
        if product:
            products.append(f"{vendor} {product}".strip())
        for v in aff.get("versions") or []:
            status = v.get("status", "")
            ver = v.get("version", "")
            lt = v.get("lessThan")
            lte = v.get("lessThanOrEqual")
            rng = f"version={ver}"
            if lt:
                rng += f" < {lt}"
            if lte:
                rng += f" <= {lte}"
            if v.get("versionType"):
                rng += f" ({v['versionType']})"
            if status:
                rng += f" [{status}]"
            lines.append(rng.strip())
    return "\n".join(lines), sorted(set(products))


def cve_to_doc(rec: dict) -> Doc | None:
    cve_meta = rec.get("cveMetadata") or {}
    cve_id = cve_meta.get("cveId")
    if not cve_id:
        return None
    state = (cve_meta.get("state") or "").upper()
    if state and state not in {"PUBLISHED", "REJECTED"}:
        return None
    cna = (rec.get("containers") or {}).get("cna") or {}
    adp = (rec.get("containers") or {}).get("adp") or []

    description = ""
    for d in cna.get("descriptions") or []:
        if d.get("lang", "").startswith("en"):
            description = _clean(d.get("value"))
            break
    if not description:
        return None

    versions_text, products = _version_facts(cna)

    severity, cvss = _cvss_of(cna.get("metrics") or [])
    if severity is None:
        for block in adp:
            severity, cvss = _cvss_of(block.get("metrics") or [])
            if severity or cvss:
                break

    weaknesses: list[str] = []
    for w in (cna.get("problemTypes") or []):
        for d in w.get("descriptions") or []:
            if d.get("description"):
                weaknesses.append(f"{d.get('type', '')} {d['description']}".strip())

    solutions = "\n".join(
        _clean(s.get("value")) for s in (cna.get("solutions") or []) if s.get("value")
    )
    workarounds = "\n".join(
        _clean(s.get("value")) for s in (cna.get("workarounds") or []) if s.get("value")
    )
    configs = "\n".join(
        _clean(s.get("value")) for s in (cna.get("configurations") or []) if s.get("value")
    )

    # 「版本区间不得切散」：把受影响版本与产品清单放进一个 section 文本里
    sections: list[str] = [f"## 漏洞描述\n{description}"]
    if products:
        sections.append("## 受影响产品\n" + "\n".join(f"- {p}" for p in products))
    if versions_text:
        sections.append("## 受影响版本区间\n" + versions_text)
    if severity or cvss is not None:
        sections.append(f"## 严重性\nCVSS 基础评分 {cvss if cvss is not None else '未提供'}，严重性等级 {severity or '未标注'}")
    if weaknesses:
        sections.append("## 弱点分类（CWE）\n" + "\n".join(f"- {w}" for w in weaknesses))
    if solutions:
        sections.append(f"## 修复建议\n{solutions}")
    if workarounds:
        sections.append(f"## 缓解措施\n{workarounds}")
    if configs:
        sections.append(f"## 配置与影响范围补充\n{configs}")

    return Doc(
        doc_id=cve_id,
        source="CVEProject/cvelistV5",
        source_type="cve_list_v5",
        title=f"{cve_id} 漏洞公告",
        text="\n\n".join(sections),
        published_at=cve_meta.get("datePublished"),
        updated_at=cve_meta.get("dateUpdated"),
        cve_id=cve_id,
        severity=severity,
        cvss=cvss,
        affected_versions=versions_text[:4000] if versions_text else None,
        meta={
            "state": state,
            "assigner": cve_meta.get("assignerShortName"),
            "products": products[:50],
        },
    )


def _cve_quality(doc: Doc) -> int:
    """CVE 记录的信息量打分：优先保留带版本区间 / CVSS / 弱点分类的记录。

    安全知识库的价值来自「影响面 + 严重性 + 修复建议」，只有一句话描述的记录
    对检索评测没有区分度（会把 recall 抬得虚高），因此在抓取阶段就按信息量排序取前 N。
    """
    score = 0
    if doc.affected_versions:
        score += 3
    if doc.cvss is not None:
        score += 2
    if doc.severity:
        score += 1
    text = doc.text
    if "## 弱点分类" in text:
        score += 1
    if "## 修复建议" in text:
        score += 2
    if len(text) >= 900:
        score += 1
    return score


def fetch_cves(limit: int, candidates: int | None = None) -> list[Doc]:
    """从 cvelistV5 取最新 CVE。

    走 GitHub 目录 API 拿到真实存在的 CVE 文件名（不猜编号、不出现 404 空转），
    再用 HTTP 线程池并发拉 raw JSON；随后按信息量排序取前 limit 篇。
    """
    from concurrent.futures import ThreadPoolExecutor

    want = candidates or max(limit * 4, 800)
    targets: list[str] = []
    # 从最新年份往下取：先 2026，再 2025，必要时 2024
    for year, n_dirs in ((2026, 4), (2025, 8), (2024, 4)):
        if len(targets) >= want:
            break
        try:
            dirs = _cve_year_dirs(year, n_dirs)
        except subprocess.CalledProcessError as exc:  # 年份目录可能不存在
            print(f"[cve] {year} 目录不可用：{exc}")
            continue
        for d in reversed(dirs):
            if len(targets) >= want:
                break
            try:
                files = _cve_files(year, d)
            except subprocess.CalledProcessError:
                continue
            for fn in reversed(files):
                targets.append(f"cves/{year}/{d}/{fn}")
            print(f"[cve] 候选 {year}/{d} -> {len(targets)}")

    parsed_docs: list[Doc] = []
    seen: set[str] = set()
    with ThreadPoolExecutor(max_workers=16) as pool:
        for rec in pool.map(_fetch_cve_json, targets):
            if not rec:
                continue
            parsed = cve_to_doc(rec)
            if parsed and parsed.doc_id not in seen:
                seen.add(parsed.doc_id)
                parsed_docs.append(parsed)
    parsed_docs.sort(key=lambda d: (-_cve_quality(d), d.doc_id))
    kept = parsed_docs[:limit]
    print(f"[cve] 解析 {len(parsed_docs)} 篇，按信息量保留前 {len(kept)} 篇")
    return kept


# ---------------------------------------------------------------- GHSA (OSV)


def _extract_members(tarball: Path, predicate, limit: int | None = None) -> list[tuple[str, bytes]]:
    """流式解包，只保留匹配的成员；limit 为 None 表示全部保留。"""
    out: list[tuple[str, bytes]] = []
    with tarfile.open(tarball, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            if predicate(member.name):
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                out.append((member.name, fh.read()))
                if limit is not None and len(out) >= limit:
                    break
    return out


def _list_members(tarball: Path, predicate) -> list[str]:
    with tarfile.open(tarball, "r:gz") as tf:
        return [m.name for m in tf if m.isfile() and predicate(m.name)]


def _read_members(tarball: Path, names: list[str]) -> list[tuple[str, bytes]]:
    wanted = set(names)
    out: list[tuple[str, bytes]] = []
    with tarfile.open(tarball, "r:gz") as tf:
        for member in tf:
            if member.isfile() and member.name in wanted:
                fh = tf.extractfile(member)
                if fh is not None:
                    out.append((member.name, fh.read()))
    return out


def ghsa_to_doc(osv: dict) -> Doc | None:
    adv_id = osv.get("id")
    if not adv_id or not adv_id.startswith("GHSA-"):
        return None
    summary = _clean(osv.get("summary"))
    details = _clean(osv.get("details"))
    if not details and not summary:
        return None
    aliases = osv.get("aliases") or []
    cve_ids = [a for a in aliases if a.startswith("CVE-")]

    severity, cvss = None, None
    for sev in osv.get("severity") or []:
        score_str = str(sev.get("score", ""))
        m = re.search(r"([0-9]+\.[0-9]+)\s*$", score_str)  # 形如 ".../E:P" 或 "9.8"
        if m:
            try:
                cvss = float(m.group(1))
            except ValueError:
                cvss = None
    db = osv.get("database_specific") or {}
    severity = db.get("severity")
    if cvss is None:
        try:
            cvss = float((db.get("cvss") or {}).get("score"))
        except (TypeError, ValueError):
            cvss = None

    pkg_lines: list[str] = []
    version_lines: list[str] = []
    for aff in osv.get("affected") or []:
        pkg = aff.get("package") or {}
        name = pkg.get("name", "")
        eco = pkg.get("ecosystem", "")
        pkg_lines.append(f"{eco}/{name}".strip("/"))
        for rng in aff.get("ranges") or []:
            events = rng.get("events") or []
            parts = []
            for ev in events:
                for k, v in ev.items():
                    parts.append(f"{k}={v}")
            if parts:
                version_lines.append(f"{name}: " + ", ".join(parts))
        for v in aff.get("versions") or []:
            version_lines.append(f"{name}: version={v}")

    sections = []
    if summary:
        sections.append(f"## 摘要\n{summary}")
    if details:
        sections.append(f"## 详情\n{details}")
    if pkg_lines:
        sections.append("## 受影响包\n" + "\n".join(f"- {p}" for p in sorted(set(pkg_lines))))
    if version_lines:
        sections.append("## 受影响版本区间\n" + "\n".join(version_lines[:200]))
    refs = [r.get("url") for r in (osv.get("references") or []) if r.get("url")]
    if refs:
        sections.append("## 参考链接\n" + "\n".join(f"- {u}" for u in refs[:10]))

    return Doc(
        doc_id=adv_id,
        source="github/advisory-database",
        source_type="ghsa",
        title=f"{adv_id} 安全公告（{(summary or details[:60]).strip()}）",
        text="\n\n".join(sections),
        published_at=_iso(osv.get("published")),
        updated_at=_iso(osv.get("modified")),
        cve_id=cve_ids[0] if cve_ids else None,
        severity=severity.lower() if isinstance(severity, str) else None,
        cvss=cvss,
        affected_versions="\n".join(version_lines[:200]) or None,
        meta={"aliases": aliases, "ecosystem": sorted({(a.get("package") or {}).get("ecosystem", "") for a in (osv.get("affected") or [])})},
    )


def _iso(ts: str | None) -> str | None:
    if not ts:
        return None
    return ts if "T" in ts else ts


def fetch_ghsa(limit: int) -> list[Doc]:
    """从 github/advisory-database 取 GHSA 公告（OSV 格式）。

    tarball 内按目录顺序排列（老 → 新），因此取**最靠后**的 limit 篇，
    即最新发布的公告——与"知识库要能增量更新"的设定一致。
    """
    tar = ADVISORY_TARBALL
    if not tar.exists():
        raise FileNotFoundError(f"缺少 {tar}，先执行抓取脚本下载 tarball")

    def predicate(n: str) -> bool:
        base = n.rsplit("/", 1)[-1]
        return (
            n.endswith(".json")
            and "/advisories/" in n
            and base.startswith("GHSA-")
            and base != "GHSA-xxxx-xxxx-xxxx.json"
        )

    names = _list_members(tar, predicate)
    print(f"[ghsa] tarball 内共 {len(names)} 篇 GHSA")
    picked = names[-max(limit * 2, limit) :]
    docs: list[Doc] = []
    seen: set[str] = set()
    # 分批读，避免一次性把所有 JSON 读进内存
    batch = 400
    for start in range(len(picked), 0, -batch):
        chunk = picked[max(0, start - batch) : start]
        for name, blob in reversed(_read_members(tar, chunk)):
            if len(docs) >= limit:
                break
            try:
                osv = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if len((osv.get("details") or "")) < 80:
                continue
            doc = ghsa_to_doc(osv)
            if doc and doc.doc_id not in seen:
                seen.add(doc.doc_id)
                docs.append(doc)
        if len(docs) >= limit:
            break
    return docs


# ---------------------------------------------------------------- MITRE ATT&CK


def attack_to_doc(obj: dict) -> Doc | None:
    if obj.get("type") != "attack-pattern":
        return None
    if obj.get("revoked") or obj.get("x_mitre_deprecated"):
        return None
    ext = {r["external_id"]: r for r in obj.get("external_references", []) if r.get("external_id")}
    tech_id = None
    for k, r in ext.items():
        if r.get("source_name") == "mitre-attack":
            tech_id = k
            break
    if not tech_id:
        return None

    name = obj.get("name", "")
    description = _clean(obj.get("description"))
    if not description:
        return None

    tactics: list[str] = []
    for phase in obj.get("kill_chain_phases") or []:
        if phase.get("kill_chain_name") == "mitre-attack":
            tactics.append(phase.get("phase_name", ""))

    platforms = obj.get("x_mitre_platforms") or []
    detection = _clean(obj.get("x_mitre_detection"))
    aliases = obj.get("x_mitre_aliases") or []

    sections = [f"## 技术点描述\n{description}"]
    if detection:
        sections.append(f"## 检测建议\n{detection}")
    if platforms:
        sections.append("## 适用平台\n" + ", ".join(platforms))
    if tactics:
        sections.append("## 战术阶段\n" + ", ".join(tactics))
    if aliases:
        sections.append("## 别名\n" + ", ".join(aliases))

    return Doc(
        doc_id=tech_id,
        source="mitre-attack/attack-stix-data",
        source_type="attack_stix",
        title=f"{tech_id} {name}（ATT&CK 技术点）",
        text="\n\n".join(sections),
        published_at=None,
        updated_at=_iso(obj.get("modified")),
        severity=None,
        cvss=None,
        affected_versions=None,
        meta={"tactics": tactics, "platforms": platforms, "attack_id": tech_id},
    )


def fetch_attack(limit: int) -> list[Doc]:
    tar = ATTACK_TARBALL
    if not tar.exists():
        raise FileNotFoundError(f"缺少 {tar}，先执行抓取脚本下载 tarball")
    members = _extract_members(
        tar,
        lambda n: n.endswith("enterprise-attack/enterprise-attack.json"),
        1,
    )
    if not members:
        raise RuntimeError("attack-stix-data tarball 中未找到 enterprise-attack.json")
    bundle = json.loads(members[0][1])
    objs = [o for o in bundle.get("objects", []) if o.get("type") == "attack-pattern"]
    # 有描述、未废弃的技术点，按 ATT&CK ID 排序后均匀取样，保证覆盖不同战术
    objs.sort(key=lambda o: str(o.get("name")))
    parsed = [d for d in (attack_to_doc(o) for o in objs) if d]
    if len(parsed) <= limit:
        return parsed
    step = len(parsed) / limit
    return [parsed[int(i * step)] for i in range(limit)]


# ---------------------------------------------------------------- OWASP Cheat Sheets


def owasp_to_doc(path: str, text: str) -> Doc | None:
    text = _clean(text)
    if len(text) < 400:
        return None
    m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = m.group(1).strip() if m else Path(path).stem
    slug = Path(path).stem
    # 去掉开头的目录节（Cheat Sheet 正文前的 link 列表），保留正文
    body = re.sub(r"^\s*\[.*?\]:.*$", "", text, flags=re.MULTILINE)
    return Doc(
        doc_id=f"owasp:{slug}",
        source="OWASP/CheatSheetSeries",
        source_type="owasp_cheatsheet",
        title=f"OWASP Cheat Sheet: {title}",
        text=body,
        published_at=None,
        updated_at=None,
        meta={"path": path, "license": "CC-BY-SA-4.0"},
    )


def fetch_owasp(limit: int) -> list[Doc]:
    tar = OWASP_TARBALL
    if not tar.exists():
        raise FileNotFoundError(f"缺少 {tar}，先执行抓取脚本下载 tarball")
    members = _extract_members(
        tar,
        lambda n: "/cheatsheets/" in n and n.endswith(".md") and "/assets/" not in n,
        limit * 3,
    )
    docs: list[Doc] = []
    for name, blob in members:
        if len(docs) >= limit:
            break
        doc = owasp_to_doc(name, blob.decode("utf-8", errors="replace"))
        if doc:
            docs.append(doc)
    return docs


# ---------------------------------------------------------------- 入口


def fetch_all() -> dict:
    manifest: dict = {"fetched_at": _now(), "sources": {}}
    plan = [
        ("cve", "cve_list_v5", fetch_cves),
        ("ghsa", "ghsa", fetch_ghsa),
        ("attack", "attack_stix", fetch_attack),
        ("owasp", "owasp_cheatsheet", fetch_owasp),
    ]
    for key, source_type, fn in plan:
        limit = CORPUS_LIMITS[key]
        t0 = time.time()
        docs = fn(limit)
        n, sha = write_jsonl(CORPUS_DIR / f"{key}.jsonl", docs)
        manifest["sources"][key] = {
            "source_type": source_type,
            "requested": limit,
            "documents": n,
            "sha256": sha,
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(f"[fetch] {key}: {n} 篇，sha256={sha[:16]}…，{time.time() - t0:.1f}s")
    total = sum(s["documents"] for s in manifest["sources"].values())
    manifest["total_documents"] = total
    (CORPUS_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[fetch] 合计 {total} 篇")
    return manifest


if __name__ == "__main__":
    fetch_all()
