## 5. 失败案例分析（Badcase 四分类归因）

评测集 100 条，失败 9 条（最终配置 D）。

| 归因类别 | 含义 | 条数 |
| --- | --- | --- |
| `retrieval_miss` | gold 完全没进候选池（分块 / 分词 / embedding） | 4 |
| `rerank_misorder` | 召回了但被挤出送入上下文的名次（重排 / 候选数） | 0 |
| `context_truncated` | 进了候选但被 token 预算裁掉（预算 / 冗余 chunk） | 5 |
| `generation_halluc` | 上下文里有答案却答错或编造（prompt / 生成模型） | 0 |

### retrieval_miss（4 条，示例最多 5 条）

- `q059` T1547.001 这个技术点是什么？它可以通过哪些启动文件夹路径和注册表键实现持久化？ ｜ gold=['T1547.001#1'] ｜ top5=['T1037.003#0', 'T1037.003#1', 'T1003.008#0', 'T1003.008#1', 'T1003.008#2'] ｜ top_score=0.5601487084916209 ｜ refused=True
- `q060` T1555.006 这个技术点是什么？攻击者如何获取凭据？检测时关注哪些命令？ ｜ gold=['T1555.006#0'] ｜ top5=['T1033#0', 'T1047#0', 'T1018#0', 'T1027.010#0', 'T1056.003#0'] ｜ top_score=0.726955596608926 ｜ refused=False
- `q064` T1593.002 这个技术点是什么？攻击者如何使用搜索引擎？检测方面原文提到了什么？ ｜ gold=['T1593.002#0'] ｜ top5=['T1033#0', 'T1018#0', 'T1056.003#0', 'T1003.008#0', 'T1003.008#1'] ｜ top_score=0.6293290061511441 ｜ refused=True
- `q067` 在 Xcode 中，为什么启用 Scribble guards、Edge guards、Malloc guards 和 Zombies 这些额外诊断工具时需要注意一个限制？这个限制是什么？ ｜ gold=['owasp:C-Based_Toolchain_Hardening_Cheat_Sheet#50'] ｜ top5=['owasp:C-Based_Toolchain_Hardening_Cheat_Sheet#57', 'owasp:C-Based_Toolchain_Hardening_Cheat_Sheet#58', 'owasp:C-Based_Toolchain_Hardening_Cheat_Sheet#5', 'owasp:C-Based_Toolchain_Hardening_Cheat_Sheet#6', 'GHSA-xwc4-2qqp-pxh3#1'] ｜ top_score=1.494386334837118 ｜ refused=False

### context_truncated（5 条，示例最多 5 条）

- `q039` 根据该 CVE 公告，哪些版本被标记为受影响（affected）？ ｜ gold=['CVE-2025-71064#2'] ｜ top5=['CVE-2026-97871#2', 'CVE-2026-96777#2', 'CVE-2026-97871#3', 'CVE-2026-96763#2', 'CVE-2026-96762#2'] ｜ top_score=1.379519475 ｜ refused=False
- `q053` 针对 CVE-2026-97724，官方给出的修复建议是什么？需要升级到哪个版本？ ｜ gold=['CVE-2026-97724#4'] ｜ top5=['GHSA-xwwx-5pjp-p9gg#1', 'GHSA-xrq6-q643-mcv3#1', 'GHSA-xrff-5qhc-2wvh#1', 'CVE-2026-97724#0', 'CVE-2026-97724#1'] ｜ top_score=0.9121762170982817 ｜ refused=False
- `q072` surrealdb surrealdb 下，CVE-2025-71390 与 CVE-2025-71392 两个漏洞各自的受影响版本区间分别是什么？ ｜ gold=['CVE-2025-71390#2', 'CVE-2025-71392#2'] ｜ top5=['CVE-2025-71390#1', 'CVE-2025-71392#1', 'CVE-2025-71392#0', 'CVE-2025-71390#0', 'CVE-2025-71397#1'] ｜ top_score=1.4776666095186806 ｜ refused=False
- `q073` Mattermost Mattermost 下，CVE-2026-96259 与 CVE-2026-96260 两个漏洞各自的受影响版本区间分别是什么？ ｜ gold=['CVE-2026-96259#2', 'CVE-2026-96260#2'] ｜ top5=['CVE-2026-96259#0', 'CVE-2026-96260#0', 'CVE-2026-96259#1', 'CVE-2026-96260#1', 'CVE-2026-96259#4'] ｜ top_score=1.393166325 ｜ refused=False
- `q074` kvcache-ai mooncake 下，CVE-2026-96763 与 CVE-2026-96762 两个漏洞各自的受影响版本区间分别是什么？ ｜ gold=['CVE-2026-96763#2', 'CVE-2026-96762#2'] ｜ top5=['CVE-2026-96762#0', 'CVE-2026-96763#0', 'CVE-2026-96762#1', 'CVE-2026-96763#1', 'CVE-2026-96764#1'] ｜ top_score=1.7615917372410412 ｜ refused=False
