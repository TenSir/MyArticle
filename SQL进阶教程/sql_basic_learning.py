import psycopg2
import pandas as pd
pd.options.display.max_columns = None

conn = psycopg2.connect(user='postgres', password='', host='localhost', port=5432, database='shop')
cursor = conn.cursor()

# 32
sql = """
CREATE TABLE Product (
	product_id CHAR(4) NOT NULL,
	product_name VARCHAR(100) NOT NULL,
	product_type VARCHAR(32) NOT NULL,
	sale_price INTEGER,
	purchase_price INTEGER,
	regist_date DATE,
	PRIMARY KEY (product_id)
	);
"""

# 37
# sql = """
# DROP TABLE Product;"""

# 38
sql = """
ALTER TABLE Product ADD COLUMN product_name_pinying VARCHAR(100)"""

# 39
sql = """
ALTER TABLE Product DROP COLUMN product_name_pinying"""

# 40
sql = """
INSERT INTO Product VALUES
	('0001', 'T恤' ,'衣服', 1000, 500, '2009-09-20'),
	('0002', '打孔器', '办公用品', 500, 320, '2009-09-11'),
	('0003', '运动T恤', '衣服', 4000, 2800, NULL),
	('0004', '菜刀', '厨房用具', 3000, 2800, '2009-09-20'),
	('0005', '高压锅', '厨房用具', 6800, 5000, '2009-01-15'),
	('0006', '叉子', '厨房用具', 500, NULL, '2009-09-20'),
	('0007', '擦菜板', '厨房用具', 880, 790, '2008-04-28'),
	('0008', '圆珠笔', '办公用品', 100, NULL, '2009-11-11');"""

# 46
sql = """
SELECT product_id, product_name, purchase_price
	FROM Product"""

# 47
sql = """
SELECT *
	FROM Product"""

# 48
sql = """
SELECT product_id AS id,
		product_name AS name,
		purchase_price AS price
	FROM Product;"""

# 49
sql = """
SELECT product_id AS "商品编号",
		product_name AS "商品名称",
		purchase_price AS "进货单价"
	FROM Product;"""

# 50
sql = """
SELECT '商品' AS string, 38 AS number, '2009-02-24' AS date,
		product_id, product_name
	FROM Product;"""

# 51
sql = """
SELECT DISTINCT product_type
	FROM Product"""

# 52
sql = """
SELECT DISTINCT purchase_price
	FROM Product"""

# 52
sql = """
SELECT DISTINCT product_type, regist_date
	FROM Product"""

# 54
sql = """
SELECT product_type, product_name
	FROM Product
	WHERE product_type = '衣服'"""


# 57
sql = """
SELECT product_name, sale_price,
		sale_price * 2 AS "sale_price_x2"
	FROM Product"""

# 60
sql = """
SELECT product_name, product_type
	FROM Product
	WHERE sale_price <> 500"""

# 61
sql = """
SELECT product_name, product_type, regist_date
	FROM Product
	WHERE regist_date < '2009-09-27'"""

# 62
sql = """
SELECT product_name, sale_price, purchase_price
	FROM Product
	WHERE sale_price-purchase_price>=500"""

# 66
sql = """
SELECT product_name, purchase_price
	FROM Product
	WHERE purchase_price is NULL"""

# 69
sql = """
SELECT product_name, product_type, sale_price
	FROM Product
	WHERE NOT sale_price >= 1000"""

# 70
sql = """
SELECT product_name, purchase_price
	FROM Product
	WHERE product_type = '厨房用具'
		OR sale_price >= 3000"""

# 74
sql = """
SELECT product_name, product_type, regist_date
	FROM Product
	WHERE product_type = '办公用品'
		AND (regist_date = '2009-09-11'
			OR regist_date = '2009-09-20');"""

# 83
sql = """
SELECT COUNT(*), COUNT(purchase_price)
	FROM Product"""

# 88
sql = """
SELECT MAX(regist_date), MIN(regist_date)
	FROM Product"""

# 89
sql = """
SELECT COUNT(DISTINCT product_type)
	FROM Product"""

# 90
sql = """
SELECT SUM(sale_price), SUM(DISTINCT sale_price)
	FROM Product"""

# 92
sql = """
SELECT product_type, COUNT(*)
	FROM Product
	GROUP BY product_type"""

# 94
sql = """
SELECT purchase_price, COUNT(*)
	FROM Product
	WHERE product_type = '衣服'
	GROUP BY purchase_price"""

# 102
sql = """
SELECT product_type, COUNT(*)
	FROM Product
	GROUP BY product_type
	HAVING COUNT(*) = 2"""

# 103
sql = """
SELECT product_type, AVG(sale_price)
	FROM Product
	GROUP BY product_type
	HAVING AVG(sale_price) >= 2500"""

# 113
sql = """
SELECT product_id AS id, sale_price AS sp
	FROM Product
	ORDER BY sp, id"""

# 114
sql = """
SELECT product_type, COUNT(*)
	FROM Product
	GROUP BY product_type
	ORDER BY COUNT(*)"""

# 120
sql = """
CREATE TABLE ProductIns
	(product_id CHAR(4) NOT NULL,
	product_name VARCHAR(100) NOT NULL,
	product_type VARCHAR(32) NOT NULL,
	sale_price INTEGER DEFAULT 0,
	purchase_price INTEGER,
	regist_date DATE,
	PRIMARY  KEY (product_id)
	);"""

sql = """
INSERT INTO ProductIns (product_id, product_name, product_type, sale_price, purchase_price, regist_date)
	VALUES ('0001', 'T恤' ,'衣服', 1000, 500, '2009-09-20')"""

# 124
sql = """
INSERT INTO productins (product_id, product_name, product_type, sale_price, purchase_price, regist_date)
	VALUES ('0007', '擦菜板', '厨房用具', DEFAULT, 790, '2009-04-28')"""

sql = """
SELECT *
	FROM ProductIns
	WHERE product_id = '0007'"""

# 126
sql = """
CREATE TABLE ProductCopy
	(product_id CHAR(4) NOT NULL,
	product_name VARCHAR(100) NOT NULL,
	product_type VARCHAR(32) NOT NULL,
	sale_price INTEGER,
	purchase_price INTEGER,
	regist_date DATE,
	PRIMARY  KEY (product_id)
	);"""

sql = """
INSERT INTO ProductCopy
	SELECT *
		FROM Product"""

sql = """
SELECT *
	FROM ProductCopy"""

# 127
sql = """
CREATE TABLE ProductType
	(product_type VARCHAR(32) NOT NULL,
	sum_sale_price INTEGER,
	sum_purchase_price INTEGER,
	PRIMARY KEY (product_type))"""

sql = """
INSERT INTO ProductType
	SELECT product_type, SUM(sale_price), SUM(purchase_price)
		FROM Product
		GROUP BY product_type"""

sql = """
SELECT *
	FROM ProductType"""

# 137
sql = """
UPDATE Product
	SET sale_price = sale_price * 10,
		purchase_price = purchase_price / 2
	WHERE product_type = '厨房用具'"""

# 140
sql = """
UPDATE Product
	SET sale_price = sale_price - 1000
	WHERE product_name = '运动T恤衫';
UPDATE Product
	SET sale_price = sale_price + 1000
	WHERE product_name = 'T恤衫';"""

# 152
sql = """
CREATE VIEW ProductSum (product_type, cnt_product)
AS
SELECT product_type, COUNT(*)
	FROM Product
	GROUP BY product_type"""

sql = """
SELECT *
	FROM ProductSum"""

# 161
sql = """
SELECT product_type, cnt_product
	FROM (SELECT product_type, COUNT(*) AS cnt_product
			FROM Product
			GROUP BY product_type) AS ProductSum"""
# HINT:  例如, FROM (SELECT ...) [AS] foo.

# 162
sql = """
SELECT product_type, cnt_product
	FROM (SELECT *
			FROM (SELECT product_type, COUNT(*) AS cnt_product
					FROM Product
					GROUP BY product_type) AS ProductSum
			WHERE cnt_product = 4) AS ProductSum2"""

# 165
sql = """
SELECT product_id, product_name, sale_price
	FROM Product
	WHERE sale_price > (SELECT AVG(sale_price) FROM Product)"""

# 166
sql = """
SELECT product_id,
		product_name,
		sale_price,
		(SELECT AVG(sale_price) FROM Product) AS avg_price
	FROM Product"""

# 166
sql = """
SELECT product_type, AVG(sale_price)
	FROM Product
	GROUP BY product_type
	HAVING AVG(sale_price) > (SELECT AVG(sale_price) FROM Product)"""

# 170
sql = """
SELECT product_type, product_name, sale_price
	FROM Product AS P1
	WHERE sale_price > (
		SELECT AVG(sale_price)
			FROM Product AS P2
			WHERE P1.product_type = p2.product_type
			GROUP BY product_type)"""

# 178
sql = """
CREATE TABLE SampleMath (
	m NUMERIC (10, 3),
	n INTEGER,
	p INTEGER )"""

sql = """
INSERT INTO SampleMath VALUES
	(500, 0, NULL),
	(-180, 0, NULL),
	(NULL, NULL, NULL),
	(NULL, 7, 3),
	(NULL, 5, 2),
	(NULL, 4, NULL),
	(8, NULL, 3),
	(2.27, 1, NULL),
	(5.555, 2, NULL),
	(NULL, 1, NULL),
	(8.76, NULL, NULL)"""

sql = """
SELECT *
	FROM SampleMath"""

# 180
sql = """
SELECT m, ABS(m) AS abs_col
	FROM SampleMath -- ||"""

# 190
sql = """
SELECT CURRENT_DATE"""

# 191
sql = """
SELECT CURRENT_TIME"""

# 192
sql = """
SELECT CURRENT_TIMESTAMP,
		CAST(EXTRACT(YEAR FROM CURRENT_TIMESTAMP) AS INTEGER) AS year"""

# 199
sql = """
CREATE TABLE SampleLike (
	strcol VARCHAR(6) NOT NULL,
	PRIMARY KEY (strcol)
	)"""

sql = """
INSERT INTO samplelike (strcol) VALUES
	('abcddd'),
	('dddabc'),
	('abdddc'),
	('abcdd'),
	('ddabc'),
	('abddc')
	"""

# 200
sql = """
SELECT *
	FROM SampleLike
	WHERE strcol LIKE '%ddd'"""

# 202
sql = """
SELECT *
	FROM SampleLike
	WHERE strcol LIKE 'abc___'"""

# 202
sql = """
SELECT product_name, sale_price
	FROM Product
	WHERE sale_price BETWEEN 100 AND 1000"""

# 205
sql = """
SELECT product_name, purchase_price
	FROM Product
	WHERE purchase_price in (320, 500, 5000)"""

# 206
sql = """
CREATE TABLE shopproduct
	(shop_id CHAR(4) NOT NULL,
	shop_name VARCHAR(200) NOT NULL,
	product_id CHAR(4) NOT NULL,
	quantity INTEGER NOT NULL,
	PRIMARY KEY (shop_id, product_id))"""

sql = """
INSERT INTO shopproduct (shop_id, shop_name, product_id, quantity) VALUES
	('000A',	'东京',		'0001',	30),
	('000A',	'东京',		'0002',	50),
	('000A',	'东京',		'0003',	15),
	('000B',	'名古屋',	'0002',	30),
	('000B',	'名古屋',	'0003',	120),
	('000B',	'名古屋',	'0004',	20),
	('000B',	'名古屋',	'0006',	10),
	('000B',	'名古屋',	'0007',	40),
	('000C',	'大阪',		'0003',	20),
	('000C',	'大阪',		'0004',	50),
	('000C',	'大阪',		'0006',	90),
	('000C',	'大阪',		'0007',	70),
	('000D',	'福冈',		'0001',	100)"""

# 208
sql = """
SELECT product_name, sale_price
	FROM Product
	WHERE product_id in (
		SELECT product_id
			FROM ShopProduct
			WHERE shop_id = '000C')"""

# 211
sql = """
SELECT product_name, sale_price
	FROM Product AS P
	WHERE EXISTS (
		SELECT *
			FROM ShopProduct AS SP
			WHERE SP.shop_id = '000C'
				AND P.product_id = SP.product_id)"""

# 216
sql = """
SELECT product_name,
		CASE WHEN product_type = '衣服' THEN 'A:' || product_type
			WHEN product_type = '办公用品' THEN 'B:' || product_type
			WHEN product_type = '厨房用具' THEN 'C:' || product_type
			ELSE NULL
		END AS abc_product_type
	FROM Product"""

# 217
sql = """
SELECT product_type, SUM(sale_price) AS sum_price
	FROM Product
	GROUP BY product_type"""

# 218
sql = """
SELECT SUM(CASE WHEN product_type = '衣服' THEN sale_price ELSE 0 END) AS sum_price_clothes,
		SUM(CASE WHEN product_type = '厨房用具' THEN sale_price ELSE 0 END) AS sum_price_kitchen,
		SUM(CASE WHEN product_type = '办公用品' THEN sale_price ELSE 0 END) AS sum_price_office
	FROM Product"""

# 226
sql = """
CREATE TABLE product2
	(product_id CHAR(4) NOT NULL,
	product_name VARCHAR(100) NOT NULL,
	product_type VARCHAR(32) NOT NULL,
	sale_price INTEGER,
	purchase_price INTEGER,
	regist_date DATE,
	PRIMARY  KEY (product_id)
	);"""

sql = """
INSERT INTO product2 VALUES
	('0001', 'T恤' ,'衣服', 1000, 500, '2009-09-20'),
	('0002', '打孔器', '办公用品', 500, 320, '2009-09-11'),
	('0003', '运动T恤', '衣服', 4000, 2800, NULL),
	('0009', '手套', '衣服', 800, 500, NULL),
	('0010', '水壶', '厨房用具', 2000, 1700, '2009-09-20');"""

# 228
sql = """
SELECT product_id, product_name
	FROM Product
-- UNION ALL
-- INTERSECT
EXCEPT
SELECT product_id, product_name
	FROM Product2"""

# 237
sql = """
SELECT SP.shop_id, SP.shop_name, SP.product_id, P.product_name
	FROM ShopProduct AS SP LEFT OUTER JOIN Product AS P
		ON SP.product_id = P.product_id"""

# 244
sql = """
CREATE TABLE inventoryproduct
(inventory_id CHAR(4) NOT NULL,
product_id CHAR(4) NOT NULL,
inventory_quantity INTEGER NOT NULL,
PRIMARY  KEY (inventory_id, product_id));

INSERT INTO inventoryproduct VALUES 
('S001',	'0001',	0),
('S001',	'0002',	120),
('S001',	'0003',	200),
('S001',	'0004',	3),
('S001',	'0005',	0),
('S001',	'0006',	99),
('S001',	'0007',	999),
('S001',	'0008',	200),
('S002',	'0001',	10),
('S002',	'0002',	25),
('S002',	'0003',	34),
('S002',	'0004',	19),
('S002',	'0005',	99),
('S002',	'0006',	0),
('S002',	'0007',	0),
('S002',	'0008',	18);"""

# 245
sql = """
SELECT sp.shop_id, sp.shop_name, sp.product_id, p.product_name, p.sale_price, ip.inventory_quantity
	FROM shopproduct AS sp INNER JOIN product AS p
		ON sp.product_id = p.product_id
		INNER JOIN inventoryproduct AS ip
			ON sp.product_id = ip.product_id
	WHERE ip.inventory_id = 'S001'"""

# 246
sql = """
SELECT SP.shop_id, SP.shop_name, SP.product_id, P.product_name
	FROM ShopProduct AS SP CROSS JOIN Product AS P"""

# 259
sql = """
SELECT product_name, product_type, sale_price,
		RANK() OVER (PARTITION BY product_type ORDER BY sale_price) AS ranking
	FROM Product"""

# 261
sql = """
SELECT product_name, product_type, sale_price,
		RANK() OVER (ORDER BY sale_price) AS ranking
	FROM Product"""

# 268
sql = """
SELECT product_id, product_name, sale_price,
		AVG(sale_price) OVER (ORDER BY product_id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS moving_avg
	FROM Product"""

# 272
sql = """
SELECT '合计' AS product_type, SUM(sale_price)
	FROM Product
UNION ALL
SELECT product_type, SUM(sale_price)
	FROM Product
	GROUP BY product_type"""

# 274
sql = """
SELECT product_type, regist_date, SUM(sale_price) AS sum_price
	FROM Product
	GROUP BY ROLLUP(product_type, regist_date)"""

# 277
sql = """
SELECT CASE WHEN GROUPING(product_type)=1 THEN '商品种类合计' ELSE product_type END AS product_type,
		CASE WHEN GROUPING(regist_date)=1 THEN '日期合计' ELSE CAST(regist_date AS VARCHAR(16)) END AS regist_date,
		SUM(sale_price) AS sum_price
	FROM Product
	-- GROUP BY ROLLUP(product_type, regist_date)
	-- GROUP BY CUBE(product_type, regist_date)
	GROUP BY GROUPING SETS (product_type, regist_date)"""


df = pd.read_sql(sql, conn)
print(df)

cursor.execute(sql)
conn.commit()
cursor.close()
conn.close()
