USE `compliance_result`;

ALTER TABLE `field_classification_result`
  ADD COLUMN `is_personal` BOOLEAN NULL AFTER `subcategory`;
