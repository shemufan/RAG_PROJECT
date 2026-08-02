USE `enterprise_source`;

INSERT INTO `employee` (
  employee_id, employee_name, id_card_no, phone, email, gender,
  birth_date, home_address, salary, department, job_title, join_date
) VALUES
  ('E0001', '演示员工甲', '11010019900101001X', '13800000001', 'employee1@example.test', '女', '1990-01-01', '虚构市测试区一号路1号', 12000.00, '研发部', '工程师', '2021-03-01'),
  ('E0002', '演示员工乙', '110100199202020022', '13800000002', 'employee2@example.test', '男', '1992-02-02', '虚构市测试区二号路2号', 13500.00, '研发部', '工程师', '2020-07-15'),
  ('E0003', '演示员工丙', '110100198803030033', '13800000003', 'employee3@example.test', '女', '1988-03-03', '虚构市测试区三号路3号', 15000.00, '财务部', '会计', '2019-11-20'),
  ('E0004', '演示员工丁', '110100199404040044', '13800000004', 'employee4@example.test', '男', '1994-04-04', '虚构市测试区四号路4号', 11000.00, '市场部', '专员', '2022-05-09'),
  ('E0005', '演示员工戊', '110100198505050055', '13800000005', 'employee5@example.test', '女', '1985-05-05', '虚构市测试区五号路5号', 18000.00, '人力资源部', '经理', '2018-01-10');

INSERT INTO `customer_account` (
  customer_id, login_name, real_name, mobile, email, password_hash,
  bank_card_no, registered_ip, device_id, account_status, created_at
) VALUES
  ('C0001', 'demo_user_1', '虚构客户甲', '13900000001', 'customer1@example.test', 'demo_hash_001', '6222020000000000001', '192.0.2.1', 'DEMO-DEVICE-001', 'ACTIVE', '2026-01-01 09:00:00'),
  ('C0002', 'demo_user_2', '虚构客户乙', '13900000002', 'customer2@example.test', 'demo_hash_002', '6222020000000000002', '192.0.2.2', 'DEMO-DEVICE-002', 'ACTIVE', '2026-01-02 09:00:00'),
  ('C0003', 'demo_user_3', '虚构客户丙', '13900000003', 'customer3@example.test', 'demo_hash_003', '6222020000000000003', '192.0.2.3', 'DEMO-DEVICE-003', 'LOCKED', '2026-01-03 09:00:00'),
  ('C0004', 'demo_user_4', '虚构客户丁', '13900000004', 'customer4@example.test', 'demo_hash_004', '6222020000000000004', '192.0.2.4', 'DEMO-DEVICE-004', 'ACTIVE', '2026-01-04 09:00:00'),
  ('C0005', 'demo_user_5', '虚构客户戊', '13900000005', 'customer5@example.test', 'demo_hash_005', '6222020000000000005', '192.0.2.5', 'DEMO-DEVICE-005', 'DISABLED', '2026-01-05 09:00:00');

INSERT INTO `customer_order` (
  order_id, customer_id, receiver_name, receiver_phone, delivery_address,
  product_name, order_amount, transaction_no, order_status, created_at
) VALUES
  ('O0001', 'C0001', '虚构收货人甲', '13700000001', '虚构市示例路101号', '演示商品A', 99.00, 'DEMO-TXN-0001', 'PAID', '2026-02-01 10:00:00'),
  ('O0002', 'C0002', '虚构收货人乙', '13700000002', '虚构市示例路102号', '演示商品B', 199.00, 'DEMO-TXN-0002', 'SHIPPED', '2026-02-02 10:00:00'),
  ('O0003', 'C0003', '虚构收货人丙', '13700000003', '虚构市示例路103号', '演示商品C', 299.00, 'DEMO-TXN-0003', 'CREATED', '2026-02-03 10:00:00'),
  ('O0004', 'C0004', '虚构收货人丁', '13700000004', '虚构市示例路104号', '演示商品D', 399.00, 'DEMO-TXN-0004', 'COMPLETED', '2026-02-04 10:00:00'),
  ('O0005', 'C0005', '虚构收货人戊', '13700000005', '虚构市示例路105号', '演示商品E', 499.00, 'DEMO-TXN-0005', 'CANCELLED', '2026-02-05 10:00:00');

INSERT INTO `product` (
  product_id, product_name, category, price, stock, supplier_name, description, created_at
) VALUES
  ('P0001', '演示键盘', '办公用品', 199.00, 120, '虚构供应商甲', '仅用于演示的机械键盘', '2026-01-01 08:00:00'),
  ('P0002', '演示鼠标', '办公用品', 99.00, 200, '虚构供应商乙', '仅用于演示的无线鼠标', '2026-01-02 08:00:00'),
  ('P0003', '演示显示器', '电子设备', 1299.00, 40, '虚构供应商丙', '仅用于演示的显示器', '2026-01-03 08:00:00'),
  ('P0004', '演示笔记本', '电子设备', 5999.00, 15, '虚构供应商丁', '仅用于演示的笔记本电脑', '2026-01-04 08:00:00'),
  ('P0005', '演示桌椅', '办公家具', 899.00, 30, '虚构供应商戊', '仅用于演示的办公桌椅', '2026-01-05 08:00:00');
