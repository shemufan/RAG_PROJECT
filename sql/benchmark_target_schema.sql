USE `compliance_result`;

CREATE TABLE IF NOT EXISTS `benchmark_run` (
  `run_id` CHAR(36) NOT NULL,
  `batch_name` VARCHAR(64) NOT NULL,
  `status` VARCHAR(32) NOT NULL,
  `total_cases` INT NOT NULL DEFAULT 0,
  `success_cases` INT NOT NULL DEFAULT 0,
  `failed_cases` INT NOT NULL DEFAULT 0,
  `tp` INT NOT NULL DEFAULT 0,
  `fp` INT NOT NULL DEFAULT 0,
  `tn` INT NOT NULL DEFAULT 0,
  `fn` INT NOT NULL DEFAULT 0,
  `precision_score` DECIMAL(10,8) NOT NULL DEFAULT 0,
  `recall_score` DECIMAL(10,8) NOT NULL DEFAULT 0,
  `f1_score` DECIMAL(10,8) NOT NULL DEFAULT 0,
  `accuracy_score` DECIMAL(10,8) NOT NULL DEFAULT 0,
  `coverage_score` DECIMAL(10,8) NOT NULL DEFAULT 0,
  `effective_recall_score` DECIMAL(10,8) NOT NULL DEFAULT 0,
  `model_name` VARCHAR(128) NOT NULL,
  `knowledge_base_version` VARCHAR(64) NOT NULL,
  `started_at` DATETIME(6) NOT NULL,
  `finished_at` DATETIME(6) NULL,
  `error_message` TEXT NULL,
  PRIMARY KEY (`run_id`),
  KEY `idx_benchmark_run_batch` (`batch_name`),
  KEY `idx_benchmark_run_started` (`started_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='字段识别评测任务';

CREATE TABLE IF NOT EXISTS `benchmark_prediction` (
  `prediction_id` BIGINT NOT NULL AUTO_INCREMENT,
  `run_id` CHAR(36) NOT NULL,
  `benchmark_id` BIGINT NOT NULL,
  `field_name_snapshot` VARCHAR(128) NOT NULL,
  `sample_values_json` JSON NOT NULL,
  `expected_personal` BOOLEAN NOT NULL,
  `predicted_personal` BOOLEAN NULL,
  `outcome` VARCHAR(16) NOT NULL,
  `category` VARCHAR(256) NULL,
  `subcategory` VARCHAR(256) NULL,
  `level` VARCHAR(8) NULL,
  `confidence` DECIMAL(10,8) NULL,
  `reason` TEXT NULL,
  `need_review` BOOLEAN NULL,
  `decision_path` VARCHAR(128) NULL,
  `evidence_json` JSON NULL,
  `status` VARCHAR(16) NOT NULL,
  `error_message` TEXT NULL,
  `created_at` DATETIME(6) NOT NULL,
  PRIMARY KEY (`prediction_id`),
  UNIQUE KEY `uq_benchmark_run_case` (`run_id`, `benchmark_id`),
  KEY `idx_benchmark_prediction_outcome` (`run_id`, `outcome`),
  CONSTRAINT `fk_benchmark_prediction_run`
    FOREIGN KEY (`run_id`) REFERENCES `benchmark_run` (`run_id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='字段识别逐案例预测';
