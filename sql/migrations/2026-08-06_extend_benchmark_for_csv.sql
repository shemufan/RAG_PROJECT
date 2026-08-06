USE `compliance_result`;

-- One-time migration for databases created before direct CSV input support.
ALTER TABLE `benchmark_run`
  ADD COLUMN `source_type` VARCHAR(32) NOT NULL DEFAULT 'mysql_benchmark' AFTER `non_personal_limit`,
  ADD COLUMN `input_mode` VARCHAR(16) NULL AFTER `source_type`,
  ADD COLUMN `source_name` VARCHAR(255) NULL AFTER `input_mode`,
  ADD COLUMN `source_fingerprint` CHAR(64) NULL AFTER `source_name`,
  ADD COLUMN `label_fingerprint` CHAR(64) NULL AFTER `source_fingerprint`,
  ADD COLUMN `labeled_cases` INT NOT NULL DEFAULT 0 AFTER `failed_cases`,
  ADD COLUMN `unlabeled_cases` INT NOT NULL DEFAULT 0 AFTER `labeled_cases`,
  MODIFY COLUMN `precision_score` DECIMAL(10,8) NULL,
  MODIFY COLUMN `recall_score` DECIMAL(10,8) NULL,
  MODIFY COLUMN `f1_score` DECIMAL(10,8) NULL,
  MODIFY COLUMN `accuracy_score` DECIMAL(10,8) NULL,
  MODIFY COLUMN `effective_recall_score` DECIMAL(10,8) NULL;

ALTER TABLE `benchmark_prediction`
  MODIFY COLUMN `expected_personal` BOOLEAN NULL;

UPDATE `benchmark_run`
SET `labeled_cases` = `total_cases`, `unlabeled_cases` = 0
WHERE `source_type` = 'mysql_benchmark';
