# connectors/file_loader.py
import logging
import pandas as pd
from core.schemas import FieldProfile

logger = logging.getLogger(__name__)

COLUMN_ALIAS_MAP: dict[str, list[str]] = {
    "field_name": [
        "英文名", "字段名", "字段英文名", "name", "field_name",
        "column_name", "COLUMN_NAME", "字段", "column", "属性名",
        "概念名称(英文)", "环节名称(英文)",
    ],
    "field_cn": [
        "中文名", "字段中文名", "中文名称", "field_cn", "cn_name",
        "名称", "概念名称(中文)", "环节名称(中文)", "员工姓名",
    ],
    "field_comment": [
        "业务描述", "描述", "说明", "注释", "备注", "字段描述",
        "description", "comment", "COLUMN_COMMENT", "remarks",
        "定义/描述", "环节描述", "定义", "解释", "含义",
    ],
    "data_type": [
        "字段类型", "数据类型", "data_type", "COLUMN_TYPE", "类型", "type", "dtype",
    ],
    "sample_values": [
        "样例值", "示例值", "样例", "例子", "sample", "example", "取值样例",
    ],
    "business_domain": [
        "业务域", "业务领域", "所属业务域", "domain", "业务板块", "所属产业类型",
    ],
    "database_name": [
        "数据库名", "数据库", "database", "db_name", "TABLE_SCHEMA",
    ],
    "table_name": [
        "表名", "表名称", "表格", "table", "table_name", "TABLE_NAME",
    ],
}

DOMAIN_KEYWORD_MAP: dict[str, list[str]] = {
    "hr": [
        "工资", "salary", "薪资", "payroll", "员工", "employee",
        "入职", "离职", "绩效", "考核", "奖金", "人事", "hr",
        "薪酬", "岗位", "部门", "职位", "招聘", "请假", "考勤",
        "劳动合同", "社保", "公积金", "津贴", "补贴", "加班",
    ],
    "finance": [
        "会计", "account", "财务", "finance", "税", "tax",
        "发票", "invoice", "报销", "预算", "核算", "应收",
        "应付", "账单", "凭证", "资产", "负债", "利润", "营收",
        "成本", "费用", "支出", "收入", "现金流", "账", "科目",
        "总账", "明细账", "借贷", "余额",
    ],
    "legal": [
        "法律", "legal", "合规", "compliance", "合同", "contract",
        "诉讼", "知识产权", "专利", "版权", "商标", "法规", "条款",
    ],
    "customer": [
        "客户", "customer", "用户", "user", "会员", "member",
        "消费者", "营销", "销售", "订单", "购买", "地址", "联系方式",
    ],
    "product": [
        "产品", "product", "商品", "库存", "inventory",
        "sku", "供应链", "物流", "品牌", "规格", "配方",
    ],
    "technical": [
        "服务器", "server", "数据库", "database", "日志", "log",
        "api", "接口", "配置", "config", "监控", "告警", "流水",
        "token", "密钥", "密码", "端口", "ip", "域名",
    ],
}


def clean_cell(value, default=None):
    if pd.isna(value):
        return default
    value = str(value).strip()
    return value if value else default


def resolve_column_mapping(df_columns: list[str]) -> dict[str, str | None]:
    """将 DataFrame 实际列名映射到 FieldProfile 概念。

    按 COLUMN_ALIAS_MAP 中定义的优先级顺序匹配，命中即停止。
    对于 field_name 额外尝试回退别名。

    Returns:
        {"field_name": "英文名", "field_comment": "描述", ...}
        未匹配到的概念值为 None。
    """
    mapping: dict[str, str | None] = {}
    for concept, aliases in COLUMN_ALIAS_MAP.items():
        for alias in aliases:
            if alias in df_columns:
                mapping[concept] = alias
                break
        else:
            mapping[concept] = None

    logger.info(
        "列映射完成: field_name=%s, field_cn=%s, field_comment=%s, "
        "data_type=%s, sample_values=%s, business_domain=%s, "
        "database_name=%s, table_name=%s",
        mapping.get("field_name"), mapping.get("field_cn"),
        mapping.get("field_comment"), mapping.get("data_type"),
        mapping.get("sample_values"), mapping.get("business_domain"),
        mapping.get("database_name"), mapping.get("table_name"),
    )

    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        logger.warning("数据中未找到以下概念的匹配列: %s", ", ".join(missing))

    return mapping


def infer_domain(field_name: str = "", field_cn: str = "",
                 field_comment: str = "") -> str:
    """根据字段名/中文名/注释中的关键词推断业务域。"""
    text = " ".join(
        (v or "") for v in [field_name, field_cn, field_comment]
    ).lower()
    for domain, keywords in DOMAIN_KEYWORD_MAP.items():
        for kw in keywords:
            if kw.lower() in text:
                return domain
    return "general"


def load_field_profiles_from_file(file_path: str):
    """从 Excel/CSV 文件加载字段画像。

    自动将文件列名映射到 FieldProfile 概念（通过 COLUMN_ALIAS_MAP）。
    当 business_domain 未在数据中显式提供时，使用关键词推断。
    当 field_name 无法解析时，跳过该行并记录警告。

    Returns:
        (df, fields): 原始 DataFrame 和 FieldProfile 列表。
    """
    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    col_map = resolve_column_mapping(list(df.columns))

    # 确定 field_name 的来源列
    fn_col = col_map.get("field_name")
    if fn_col is None:
        logger.error("无法找到字段名的匹配列，文件可能不是字段元数据表: %s", file_path)
        return df, []

    fields = []

    for idx, row in df.iterrows():
        raw_fn = clean_cell(row.get(fn_col))
        if raw_fn is None:
            logger.warning("第 %d 行缺少字段名 (field_name)，已跳过", idx + 2)
            continue

        # 样例值: 优先从列读取，否则为空
        sv_col = col_map.get("sample_values")
        sample_raw = clean_cell(row.get(sv_col), "") if sv_col else ""
        sample_values = [
            v.strip() for v in sample_raw.split(";")
            if v.strip() and v.strip().lower() != "nan"
        ]

        # 读取各字段
        fc_col = col_map.get("field_cn")
        field_cn = clean_cell(row.get(fc_col)) if fc_col else None

        fcmt_col = col_map.get("field_comment")
        field_comment = clean_cell(row.get(fcmt_col)) if fcmt_col else None

        dt_col = col_map.get("data_type")
        data_type = clean_cell(row.get(dt_col)) if dt_col else None

        db_col = col_map.get("database_name")
        database_name = clean_cell(row.get(db_col)) if db_col else None

        tbl_col = col_map.get("table_name")
        table_name = clean_cell(row.get(tbl_col)) if tbl_col else None

        # 业务域: 优先从数据读取，否则推断
        bd_col = col_map.get("business_domain")
        raw_domain = clean_cell(row.get(bd_col)) if bd_col else None
        if raw_domain:
            business_domain = raw_domain
        else:
            business_domain = infer_domain(raw_fn, field_cn or "", field_comment or "")

        field = FieldProfile(
            database_name=database_name,
            table_name=table_name,
            field_name=raw_fn,
            field_cn=field_cn,
            field_comment=field_comment,
            data_type=data_type,
            sample_values=sample_values,
            business_domain=business_domain,
        )
        fields.append(field)

    logger.info("成功加载 %d 个字段 (来源: %s)", len(fields), file_path)
    return df, fields
