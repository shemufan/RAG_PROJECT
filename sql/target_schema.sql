CREATE DATABASE IF NOT EXISTS `compliance_result`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE `compliance_result`;

CREATE TABLE IF NOT EXISTS `classification_run` (
  `run_id` CHAR(36) NOT NULL,
  `source_system` VARCHAR(64) NOT NULL,
  `source_database` VARCHAR(256) NOT NULL,
  `status` VARCHAR(32) NOT NULL,
  `total_fields` INT NOT NULL DEFAULT 0,
  `success_fields` INT NOT NULL DEFAULT 0,
  `review_fields` INT NOT NULL DEFAULT 0,
  `failed_fields` INT NOT NULL DEFAULT 0,
  `model_name` VARCHAR(128) NOT NULL,
  `knowledge_base_version` VARCHAR(64) NOT NULL,
  `started_at` DATETIME(6) NOT NULL,
  `finished_at` DATETIME(6) NULL,
  `error_message` TEXT NULL,
  PRIMARY KEY (`run_id`),
  CONSTRAINT `chk_run_status`
    CHECK (`status` IN ('RUNNING', 'SUCCESS', 'PARTIAL_FAILED', 'FAILED')),
  KEY `idx_run_status` (`status`),
  KEY `idx_run_started_at` (`started_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='分类任务';

CREATE TABLE IF NOT EXISTS `data_field_asset` (
  `field_id` CHAR(36) NOT NULL,
  `source_system` VARCHAR(64) NOT NULL,
  `database_name` VARCHAR(256) NOT NULL,
  `table_name` VARCHAR(256) NOT NULL,
  `table_comment` TEXT NULL,
  `column_name` VARCHAR(256) NOT NULL,
  `column_comment` TEXT NULL,
  `data_type` VARCHAR(256) NOT NULL,
  `is_nullable` BOOLEAN NOT NULL,
  `column_key` VARCHAR(64) NULL,
  `business_domain` VARCHAR(256) NOT NULL,
  `first_seen_at` DATETIME(6) NOT NULL,
  `last_seen_at` DATETIME(6) NOT NULL,
  PRIMARY KEY (`field_id`),
  UNIQUE KEY `uq_field_identity`
    (`source_system`, `database_name`, `table_name`, `column_name`),
  KEY `idx_asset_database_table` (`database_name`, `table_name`),
  KEY `idx_asset_business_domain` (`business_domain`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='字段资产';

CREATE TABLE IF NOT EXISTS `field_classification_result` (
  `result_id` BIGINT NOT NULL AUTO_INCREMENT,
  `run_id` CHAR(36) NOT NULL,
  `field_id` CHAR(36) NOT NULL,
  `category` VARCHAR(256) NOT NULL,
  `subcategory` VARCHAR(256) NULL,
  `level` VARCHAR(2) NOT NULL,
  `confidence` DECIMAL(6, 5) NOT NULL,
  `reason` TEXT NOT NULL,
  `need_review` BOOLEAN NOT NULL,
  `decision_path` VARCHAR(128) NOT NULL,
  `input_snapshot_json` JSON NOT NULL,
  `raw_output_json` JSON NOT NULL,
  `created_at` DATETIME(6) NOT NULL,
  PRIMARY KEY (`result_id`),
  UNIQUE KEY `uq_run_field` (`run_id`, `field_id`),
  KEY `idx_result_level` (`level`),
  KEY `idx_result_category` (`category`),
  KEY `idx_result_need_review` (`need_review`),
  KEY `idx_result_field_id` (`field_id`),
  CONSTRAINT `fk_result_run`
    FOREIGN KEY (`run_id`) REFERENCES `classification_run` (`run_id`),
  CONSTRAINT `fk_result_field`
    FOREIGN KEY (`field_id`) REFERENCES `data_field_asset` (`field_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='字段分类结果';

CREATE TABLE IF NOT EXISTS `classification_evidence` (
  `evidence_id` BIGINT NOT NULL AUTO_INCREMENT,
  `result_id` BIGINT NOT NULL,
  `rank_no` INT NOT NULL,
  `document_name` VARCHAR(512) NOT NULL,
  `article` VARCHAR(256) NULL,
  `content` TEXT NOT NULL,
  `relevance_score` DECIMAL(6, 5) NULL,
  `chunk_id` VARCHAR(256) NULL,
  PRIMARY KEY (`evidence_id`),
  KEY `idx_evidence_result_id` (`result_id`),
  KEY `idx_evidence_document_name` (`document_name`),
  CONSTRAINT `fk_evidence_result`
    FOREIGN KEY (`result_id`) REFERENCES `field_classification_result` (`result_id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='分类法规依据';
