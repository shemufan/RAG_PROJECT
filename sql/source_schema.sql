CREATE DATABASE IF NOT EXISTS `enterprise_source`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE `enterprise_source`;

CREATE TABLE IF NOT EXISTS `employee` (
  `employee_id` VARCHAR(32) NOT NULL COMMENT '员工内部编号',
  `employee_name` VARCHAR(64) NOT NULL COMMENT '员工姓名',
  `id_card_no` VARCHAR(18) NOT NULL COMMENT '员工身份证号码',
  `phone` VARCHAR(20) NULL COMMENT '员工联系电话',
  `email` VARCHAR(128) NULL COMMENT '员工工作邮箱',
  `gender` VARCHAR(16) NULL COMMENT '员工性别',
  `birth_date` DATE NULL COMMENT '出生日期',
  `home_address` VARCHAR(255) NULL COMMENT '家庭住址',
  `salary` DECIMAL(12, 2) NULL COMMENT '税前月工资',
  `department` VARCHAR(64) NULL COMMENT '所属部门',
  `job_title` VARCHAR(64) NULL COMMENT '职位名称',
  `join_date` DATE NULL COMMENT '入职日期',
  PRIMARY KEY (`employee_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='员工信息';

CREATE TABLE IF NOT EXISTS `customer_account` (
  `customer_id` VARCHAR(32) NOT NULL COMMENT '客户内部编号',
  `login_name` VARCHAR(64) NOT NULL COMMENT '登录账号',
  `real_name` VARCHAR(64) NULL COMMENT '客户真实姓名',
  `mobile` VARCHAR(20) NULL COMMENT '手机号',
  `email` VARCHAR(128) NULL COMMENT '电子邮箱',
  `password_hash` VARCHAR(255) NOT NULL COMMENT '登录密码哈希',
  `bank_card_no` VARCHAR(32) NULL COMMENT '绑定银行卡号',
  `registered_ip` VARCHAR(45) NULL COMMENT '注册IP地址',
  `device_id` VARCHAR(128) NULL COMMENT '登录设备编号',
  `account_status` VARCHAR(32) NOT NULL COMMENT '账户状态',
  `created_at` DATETIME NOT NULL COMMENT '创建时间',
  PRIMARY KEY (`customer_id`),
  UNIQUE KEY `uq_customer_login_name` (`login_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户账户';

CREATE TABLE IF NOT EXISTS `customer_order` (
  `order_id` VARCHAR(32) NOT NULL COMMENT '订单编号',
  `customer_id` VARCHAR(32) NOT NULL COMMENT '客户编号',
  `receiver_name` VARCHAR(64) NOT NULL COMMENT '收货人姓名',
  `receiver_phone` VARCHAR(20) NOT NULL COMMENT '收货人联系电话',
  `delivery_address` VARCHAR(255) NOT NULL COMMENT '配送地址',
  `product_name` VARCHAR(128) NOT NULL COMMENT '商品名称',
  `order_amount` DECIMAL(12, 2) NOT NULL COMMENT '订单金额',
  `transaction_no` VARCHAR(64) NULL COMMENT '支付交易流水号',
  `order_status` VARCHAR(32) NOT NULL COMMENT '订单状态',
  `created_at` DATETIME NOT NULL COMMENT '订单创建时间',
  PRIMARY KEY (`order_id`),
  KEY `idx_order_customer_id` (`customer_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户订单';

CREATE TABLE IF NOT EXISTS `product` (
  `product_id` VARCHAR(32) NOT NULL COMMENT '商品编号',
  `product_name` VARCHAR(128) NOT NULL COMMENT '商品名称',
  `category` VARCHAR(64) NOT NULL COMMENT '商品类别',
  `price` DECIMAL(12, 2) NOT NULL COMMENT '商品价格',
  `stock` INT NOT NULL COMMENT '库存数量',
  `supplier_name` VARCHAR(128) NULL COMMENT '供应商名称',
  `description` VARCHAR(500) NULL COMMENT '商品描述',
  `created_at` DATETIME NOT NULL COMMENT '创建时间',
  PRIMARY KEY (`product_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='商品信息';
