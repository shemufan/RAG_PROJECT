-- 智能合规定级引擎 — 结果持久化表结构
-- 使用方法: mysql -u root -p <database> < db/init.sql

CREATE TABLE IF NOT EXISTS classification_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    field_name VARCHAR(255) NOT NULL COMMENT '字段英文名',
    database_name VARCHAR(255) DEFAULT NULL COMMENT '来源数据库',
    table_name VARCHAR(255) DEFAULT NULL COMMENT '来源表',
    category VARCHAR(100) NOT NULL COMMENT '数据分类',
    subcategory VARCHAR(255) DEFAULT NULL COMMENT '细分类',
    level VARCHAR(10) NOT NULL COMMENT '敏感等级 L1/L2/L3/L4/未知',
    confidence FLOAT DEFAULT 0.0 COMMENT '置信度 0.0-1.0',
    reason TEXT COMMENT '推理理由',
    evidence_json JSON COMMENT '检索依据 Evidence[]',
    need_review BOOLEAN DEFAULT FALSE COMMENT '是否需要人工复核',
    decision_path VARCHAR(50) DEFAULT 'rag_llm' COMMENT '决策路径',
    source_system VARCHAR(50) DEFAULT NULL COMMENT '数据来源 mysql/excel/manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '记录时间',

    INDEX idx_field_name (field_name),
    INDEX idx_level (level),
    INDEX idx_need_review (need_review),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='字段分类定级结果';

CREATE TABLE IF NOT EXISTS error_log_rules (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    field_name VARCHAR(255) NOT NULL COMMENT '字段英文名',
    level VARCHAR(10) NOT NULL COMMENT '标准答案 L1-L4',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',

    UNIQUE KEY uk_field_name (field_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='错题本规则持久化';
