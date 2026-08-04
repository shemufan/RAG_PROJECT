USE `enterprise_source`;

CREATE TABLE IF NOT EXISTS `benchmark_field_input` (
  `benchmark_id` BIGINT NOT NULL AUTO_INCREMENT,
  `batch_name` VARCHAR(64) NOT NULL,
  `source_dataset` VARCHAR(32) NOT NULL,
  `source_row_number` INT NOT NULL,
  `field_name` VARCHAR(128) NOT NULL,
  `sample_values_json` JSON NOT NULL,
  `expected_personal` BOOLEAN NOT NULL,
  `created_at` DATETIME(6) NOT NULL,
  PRIMARY KEY (`benchmark_id`),
  UNIQUE KEY `uq_benchmark_source_row`
    (`batch_name`, `source_dataset`, `source_row_number`),
  KEY `idx_benchmark_batch_label` (`batch_name`, `expected_personal`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='字段识别评测输入';
