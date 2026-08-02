USE `compliance_result`;

-- Query 1: 查询某次任务的所有 L3/L4 字段
SELECT a.database_name, a.table_name, a.column_name, r.category, r.level, r.confidence
FROM field_classification_result AS r
JOIN data_field_asset AS a ON a.field_id = r.field_id
WHERE r.run_id = @run_id
  AND r.level IN ('L3', 'L4')
ORDER BY a.database_name, a.table_name, a.column_name;

-- Query 2: 查询 employee 表所有字段分类
SELECT a.column_name, a.column_comment, r.category, r.subcategory, r.level, r.confidence
FROM field_classification_result AS r
JOIN data_field_asset AS a ON a.field_id = r.field_id
WHERE a.database_name = 'enterprise_source'
  AND a.table_name = 'employee'
  AND r.run_id = @run_id
ORDER BY a.column_name;

-- Query 3: 按 level 统计字段数量
SELECT r.level, COUNT(*) AS field_count
FROM field_classification_result AS r
WHERE r.run_id = @run_id
GROUP BY r.level
ORDER BY r.level;

-- Query 4: 按 category 统计字段数量
SELECT r.category, COUNT(*) AS field_count
FROM field_classification_result AS r
WHERE r.run_id = @run_id
GROUP BY r.category
ORDER BY field_count DESC, r.category;

-- Query 5: 查询需要人工复核的字段
SELECT a.database_name, a.table_name, a.column_name, r.level, r.confidence, r.reason
FROM field_classification_result AS r
JOIN data_field_asset AS a ON a.field_id = r.field_id
WHERE r.run_id = @run_id
  AND r.need_review = TRUE
ORDER BY r.confidence, a.table_name, a.column_name;

-- Query 6: 查询置信度低于 0.75 的字段
SELECT a.database_name, a.table_name, a.column_name, r.category, r.level, r.confidence
FROM field_classification_result AS r
JOIN data_field_asset AS a ON a.field_id = r.field_id
WHERE r.run_id = @run_id
  AND r.confidence < 0.75
ORDER BY r.confidence;

-- Query 7: 查询某字段引用的法规依据
SELECT e.rank_no, e.document_name, e.article, e.content, e.relevance_score, e.chunk_id
FROM classification_evidence AS e
JOIN field_classification_result AS r ON r.result_id = e.result_id
JOIN data_field_asset AS a ON a.field_id = r.field_id
WHERE a.database_name = @database_name
  AND a.table_name = @table_name
  AND a.column_name = @column_name
  AND r.run_id = @run_id
ORDER BY e.rank_no;

-- Query 8: 查询最近一次成功任务
SELECT cr.*
FROM classification_run AS cr
WHERE cr.status = 'SUCCESS'
ORDER BY cr.started_at DESC
LIMIT 1;

-- Query 9: 查询每张表的高敏感字段数量
SELECT a.database_name, a.table_name, COUNT(*) AS high_sensitive_field_count
FROM field_classification_result AS r
JOIN data_field_asset AS a ON a.field_id = r.field_id
WHERE r.run_id = @run_id
  AND r.level IN ('L3', 'L4')
GROUP BY a.database_name, a.table_name
ORDER BY high_sensitive_field_count DESC, a.table_name;

-- Query 10: 查询某业务域中的全部分类结果
SELECT a.database_name, a.table_name, a.column_name, r.category, r.level, r.confidence
FROM field_classification_result AS r
JOIN data_field_asset AS a ON a.field_id = r.field_id
WHERE r.run_id = @run_id
  AND a.business_domain = @business_domain
ORDER BY a.table_name, a.column_name;
