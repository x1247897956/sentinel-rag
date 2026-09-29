## 5. 失败案例分析（Badcase 四分类归因）

评测集 100 条，失败 42 条（最终配置 D）。

| 归因类别 | 含义 | 条数 |
| --- | --- | --- |
| `retrieval_miss` | gold 完全没进候选池（分块 / 分词 / embedding） | 4 |
| `rerank_misorder` | 召回了但被挤出送入上下文的名次（重排 / 候选数） | 0 |
| `context_truncated` | 进了候选但被 token 预算裁掉（预算 / 冗余 chunk） | 38 |
| `generation_halluc` | 上下文里有答案却答错或编造（prompt / 生成模型） | 0 |

### retrieval_miss（4 条，示例最多 5 条）

- `q059` T1547.001 这个技术点是什么？它可以通过哪些启动文件夹路径和注册表键实现持久化？ ｜ gold=['T1547.001#1'] ｜ top5=['T1037.003#2', 'T1047#3', 'T1027.010#3', 'T1036.010#2', 'T1037.003#1'] ｜ top_score=0.505526978772293 ｜ refused=True
- `q060` T1555.006 这个技术点是什么？攻击者如何获取凭据？检测时关注哪些命令？ ｜ gold=['T1555.006#0'] ｜ top5=['T1056.003#2', 'T1003.008#2', 'T1047#1', 'T1003.008#1', 'T1020.001#2'] ｜ top_score=0.5953880642129815 ｜ refused=True
- `q064` T1593.002 这个技术点是什么？攻击者如何使用搜索引擎？检测方面原文提到了什么？ ｜ gold=['T1593.002#0'] ｜ top5=['T1020.001#0', 'T1003.008#1', 'T1020.001#2', 'T1003.008#2', 'T1056.003#2'] ｜ top_score=0.5023998079945025 ｜ refused=True
- `q067` 在 Xcode 中，为什么启用 Scribble guards、Edge guards、Malloc guards 和 Zombies 这些额外诊断工具时需要注意一个限制？这个限制是什么？ ｜ gold=['owasp:C-Based_Toolchain_Hardening_Cheat_Sheet#50'] ｜ top5=['owasp:C-Based_Toolchain_Hardening_Cheat_Sheet#58', 'owasp:C-Based_Toolchain_Hardening_Cheat_Sheet#57', 'owasp:Denial_of_Service_Cheat_Sheet#15', 'owasp:Business_Logic_Security_Cheat_Sheet#12', 'owasp:C-Based_Toolchain_Hardening_Cheat_Sheet#49'] ｜ top_score=0.7297428540753274 ｜ refused=False

### context_truncated（38 条，示例最多 5 条）

- `q001` CVE-2025-70029 影响的产品及版本是什么？该漏洞的严重性如何？修复建议是什么？ ｜ gold=['CVE-2025-70029#0'] ｜ top5=['CVE-2025-70023#2', 'CVE-2025-70024#2', 'CVE-2025-70027#2', 'CVE-2025-70028#2', 'CVE-2025-70025#2'] ｜ top_score=0.5549121659135497 ｜ refused=True
- `q002` CVE-2025-70032 涉及哪个 CWE 编号、影响哪个产品及版本？ ｜ gold=['CVE-2025-70032#0'] ｜ top5=['CVE-2025-70082#3', 'CVE-2025-70032#2', 'CVE-2025-70040#0', 'CVE-2025-70820#3', 'CVE-2025-70974#3'] ｜ top_score=0.5106080523919986 ｜ refused=True
- `q003` CVE-2025-70033 涉及哪个 CWE 编号、哪个受影响产品及版本？ ｜ gold=['CVE-2025-70033#0'] ｜ top5=['CVE-2025-70082#3', 'CVE-2025-70033#2', 'CVE-2025-70820#3', 'CVE-2025-70974#3', 'CVE-2025-71063#3'] ｜ top_score=0.5199304139882609 ｜ refused=True
- `q004` CVE-2025-70060 漏洞公告中，该漏洞涉及哪个 CWE 编号、影响哪个产品及版本？ ｜ gold=['CVE-2025-70060#0'] ｜ top5=['CVE-2025-70082#3', 'CVE-2025-70047#0', 'CVE-2025-70023#0', 'CVE-2025-70046#0', 'CVE-2025-70042#0'] ｜ top_score=0.5283057084654712 ｜ refused=True
- `q005` CVE-2025-70070 影响哪个产品及版本，攻击者可通过什么方式造成什么后果？ ｜ gold=['CVE-2025-70070#0'] ｜ top5=['CVE-2025-71211#0', 'CVE-2025-70024#2', 'CVE-2025-70025#2', 'CVE-2025-70023#2', 'CVE-2025-70027#2'] ｜ top_score=0.525107215155971 ｜ refused=True
