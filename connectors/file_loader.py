# connectors/file_loader.py
import pandas as pd
from core.schemas import FieldProfile


def load_field_profiles_from_file(file_path: str):
    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    fields = []

    for _, row in df.iterrows():
        field = FieldProfile(
            database_name=row.get("数据库名", None),
            table_name=row.get("表名", None),
            field_name=row.get("英文名") or row.get("字段名"),
            field_cn=row.get("中文名", None),
            field_comment=row.get("业务描述", None),
            data_type=row.get("字段类型", None),
            sample_values=str(row.get("样例值", "")).split(";") if "样例值" in df.columns else [],
            business_domain=row.get("业务域", "general"),
        )
        fields.append(field)

    return df, fields
