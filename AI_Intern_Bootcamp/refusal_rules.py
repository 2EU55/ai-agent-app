import re


def should_refuse_by_score(score_mode: str, best_score: float | None, threshold: float) -> bool:
    if best_score is None:
        return True
    if score_mode == "relevance":
        return best_score < threshold
    if score_mode == "distance":
        return best_score > threshold
    return False


def extract_query_terms(question: str) -> list[str]:
    q = (question or "").strip()
    if not q:
        return []
    raw_terms = re.findall(r"[\u4e00-\u9fff]{2,6}", q)
    stop = {
        "公司",
        "员工",
        "政策",
        "制度",
        "规定",
        "是什么",
        "怎么",
        "如何",
        "多少",
        "几个",
        "几",
        "吗",
        "呢",
        "的",
        "一下",
        "是否",
        "可以",
        "提供",
        "标准",
        "上限",
        "期间",
        "一般",
        "通常",
        "什么",
        "上班",
    }
    seen: set[str] = set()
    terms: list[str] = []

    def add_term(t: str) -> None:
        if not t:
            return
        if t in stop:
            return
        if t in seen:
            return
        seen.add(t)
        terms.append(t)

    for seg in re.findall(r"[\u4e00-\u9fff]+", q):
        if len(seg) < 2:
            continue
        for i in range(0, len(seg) - 1):
            add_term(seg[i : i + 2])

    for t in raw_terms:
        add_term(t)
        for n in (4, 3):
            if len(t) < n:
                continue
            for i in range(0, len(t) - n + 1):
                add_term(t[i : i + n])

    return terms[:12]


def term_overlap_hits(question: str, context: str) -> list[str]:
    ctx = context or ""
    hits: list[str] = []
    for t in extract_query_terms(question):
        if t and (t in ctx):
            hits.append(t)
    return hits


def detect_missing_fields(question: str, context: str) -> str | None:
    q = (question or "").strip()
    ctx = context or ""
    if not q:
        return None

    if re.search(r"(体检)", q):
        if "体检" not in ctx:
            return "“体检政策（制度字段）”"

    if re.search(r"(远程办公|远程|在家办公|居家办公)", q):
        if not re.search(r"(远程办公|在家办公|居家办公)", ctx):
            return "“远程办公（制度字段）”"

    if re.search(r"(下午茶|零食|咖啡|茶歇)", q):
        if not re.search(r"(下午茶|零食|咖啡|茶歇)", ctx):
            return "“下午茶/茶歇（福利字段）”"

    if re.search(r"(宠物)", q):
        if "宠物" not in ctx:
            return "“宠物政策（制度字段）”"

    if re.search(r"(交通班车|通勤班车|通勤车|班车)", q):
        if not re.search(r"(交通班车|通勤班车|通勤车|班车)", ctx):
            return "“交通班车（福利字段）”"

    if re.search(r"(法定节假日|节假日)", q) and re.search(r"(加班|补偿|调休)", q):
        if not re.search(r"(法定节假日|节假日)", ctx):
            return "“节假日加班补偿（制度字段）”"
        if not re.search(
            r"(法定节假日|节假日).{0,40}(?:\d+(?:\.\d+)?|[一二三四五六七八九十]+)\s*(?:倍|%|元|小时|天)",
            ctx,
        ) and not re.search(r"(三倍|3\s*倍)", ctx):
            return "“节假日加班补偿（制度字段）”"

    if re.search(r"(加班费|加班工资|加班补偿|调休)", q):
        if not re.search(r"(加班|调休)", ctx) or not re.search(
            r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十]+)\s*(?:倍|%|元|小时|天)", ctx
        ):
            return "“加班费/调休（制度字段）”"

    if re.search(r"(迟到|早退|考勤|打卡|扣款|扣钱|罚款)", q):
        if not re.search(r"(迟到|早退|考勤|打卡|扣款|罚款)", ctx) or not re.search(
            r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十]+)\s*(?:元|次|分钟|小时|天|%|倍)", ctx
        ):
            return "“考勤扣款（制度字段）”"

    if re.search(r"(晋升|升职|职级|职等|绩效|评估)", q):
        if not re.search(r"(晋升|职级|绩效|评估)", ctx):
            return "“晋升/绩效（制度字段）”"

    if re.search(r"(绩效|评估|考核)", q) and re.search(r"(A\\s*/\\s*B\\s*/\\s*C|A/B/C|A档|B档|C档|分几档|几档|分档)", q):
        if not re.search(r"(A\\s*档|B\\s*档|C\\s*档|A\\s*/\\s*B\\s*/\\s*C|A/B/C|S\\s*档|D\\s*档)", ctx):
            return "“绩效分档（制度字段）”"

    if re.search(r"(公积金|社保|五险|商业保险|补充保险|医保|保险)", q):
        if not re.search(r"(公积金|社保|五险|商业保险|补充保险|医保|保险)", ctx):
            return "“社保/公积金/保险（制度字段）”"

    if re.search(r"(地址|在哪|哪里|位置|地点)", q):
        if not re.search(r"\d+\s*号", ctx) and not (
            re.search(r"(路|街|大道|巷|号|大厦|园区|楼|室)", ctx) and re.search(r"\d+", ctx)
        ):
            return "“地址/地点（地点字段）”"

    if re.search(r"(多少钱|多少元|金额|标准|上限|补贴|报销|费用)", q):
        if not re.search(r"(?:\d+(?:\.\d+)?)\s*(?:元|￥|%|万|千)", ctx):
            return "“金额/标准（金额字段）”"

    if re.search(r"(几点|几点钟)", q):
        if not re.search(r"(?:\d{1,2}\s*(?:点|:)\s*\d{0,2})", ctx):
            return "“到账时间点（时间字段）”"

    if re.search(r"(一个月|每月).*(总共|合计)", q) and re.search(r"(补贴|报销)", q):
        if not re.search(r"每月.{0,10}(?:餐饮)?补贴.{0,10}(?:\d+|[一二三四五六七八九十]+)\s*元", ctx):
            return "“每月补贴总额”"

    if re.search(r"(年假).*(未休|没休|未用|剩余).*(折现|兑现|提现|现金|补偿|结算|多少钱)", q):
        if not re.search(r"(折现|兑现|补偿|结算|现金)", ctx):
            return "“年假折现/补偿（制度字段）”"

    if re.search(r"(病假).*(最多|上限|最长期|最长).*(几|多少)\s*天", q) or (
        ("病假" in q) and ("最多" in q) and re.search(r"(几|多少)\s*天", q)
    ):
        if not re.search(r"病假.{0,20}(?:\d+|[一二三四五六七八九十]+)\s*天", ctx):
            return "“病假天数上限（制度字段）”"

    if re.search(r"(住宿).*(早餐|含早|早饭)", q):
        if not re.search(r"(早餐|含早|早饭)", ctx):
            return "“住宿是否含早餐（字段）”"

    if re.search(r"(育儿假|带薪育儿假|育婴假|陪产假|产假|哺乳假)", q):
        if not re.search(r"(育儿假|育婴假|陪产假|产假|哺乳假)", ctx):
            return "“育儿/产假（制度字段）”"

    if re.search(r"(多久|多长|期限|截止|什么时候|何时|试用期)", q):
        if not re.search(r"(?:\d+|[一二三四五六七八九十]+)\s*(?:天|周|月|年)", ctx) and not re.search(
            r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}", ctx
        ):
            return "“时间/时长（时间字段）”"

    if re.search(r"(几|多少)\s*个?\s*月", q):
        if not re.search(r"(?:\d+|[一二三四五六七八九十]+)\s*个?\s*月", ctx):
            return "“几个月（数量）”"

    return None
