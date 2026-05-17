# connectors/mysql_connector.py

from sqlalchemy import create_engine, text
from core.schemas import FieldProfile


def create_mysql_engine(host, port, user, password, database):
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
    return create_engine(url)


def scan_mysql_schema(engine, database_name: str, sample_limit: int = 5):
    sql = text("""
        SELECT 
            TABLE_SCHEMA,
            TABLE_NAME,
            COLUMN_NAME,
            COLUMN_TYPE,
            COLUMN_COMMENT,
            IS_NULLABLE,
            COLUMN_KEY
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = :database_name
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)
    fields = []

    with engine.connect() as conn:
        rows = conn.execute(sql, {"database_name": database_name}).mappings().all()

        for row in rows:
            sample_values = []

            # Layer1 可以先不取样例值；如果要取，注意脱敏
            try:
                sample_sql = text(
                    f"SELECT `{row['COLUMN_NAME']}` FROM `{row['TABLE_SCHEMA']}`.`{row['TABLE_NAME']}` "
                    f"WHERE `{row['COLUMN_NAME']}` IS NOT NULL LIMIT {sample_limit}"
                )
                sample_rows = conn.execute(sample_sql).fetchall()
                sample_values = [str(r[0])[:50] for r in sample_rows]
            except Exception:
                sample_values = []

            fields.append(
                FieldProfile(
                    database_name=row["TABLE_SCHEMA"],
                    table_name=row["TABLE_NAME"],
                    field_name=row["COLUMN_NAME"],
                    field_comment=row["COLUMN_COMMENT"],
                    data_type=row["COLUMN_TYPE"],
                    sample_values=sample_values,
                    business_domain="general",
                    source_system="mysql",
                )
            )

    return fields
