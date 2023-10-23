import psycopg2
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

conn = psycopg2.connect(host="localhost", user="postgres", password="", port=5432, database="sql_advanced")
cursor = conn.cursor()

# 4
sql = """
CREATE TABLE poptbl (
	pref_name VARCHAR(32) PRIMARY KEY,
	population INTEGER NOT NULL);
INSERT INTO poptbl VALUES
	('德岛', 100),
	('香川', 200),
	('爱媛', 150),
	('高知', 200),
	('福冈', 300),
	('佐贺', 100),
	('长崎', 200),
	('东京', 400),
	('群马', 50);"""

# 4
sql = """
SELECT CASE WHEN pref_name in ('德岛', '香川', '爱媛', '高知') THEN '九州'
			WHEN pref_name in ('福冈', '佐贺', '长崎') THEN '四国'
			ELSE '其他'  END AS district,
		SUM(population)
	FROM poptbl
	GROUP BY district;"""

# 7
sql = """
CREATE TABLE poptbl2
(pref_name VARCHAR(32),
sex CHAR(1) NOT NULL,
population INTEGER NOT NULL,
PRIMARY KEY(pref_name, sex));
INSERT INTO poptbl2 VALUES('德岛', '1',	60 );
INSERT INTO poptbl2 VALUES('德岛', '2',	40 );
INSERT INTO poptbl2 VALUES('香川', '1',	100);
INSERT INTO poptbl2 VALUES('香川', '2',	100);
INSERT INTO poptbl2 VALUES('爱媛', '1',	100);
INSERT INTO poptbl2 VALUES('爱媛', '2',	50 );
INSERT INTO poptbl2 VALUES('高知', '1',	100);
INSERT INTO poptbl2 VALUES('高知', '2',	100);
INSERT INTO poptbl2 VALUES('福冈', '1',	100);
INSERT INTO poptbl2 VALUES('福冈', '2',	200);
INSERT INTO poptbl2 VALUES('佐贺', '1',	20 );
INSERT INTO poptbl2 VALUES('佐贺', '2',	80 );
INSERT INTO poptbl2 VALUES('长崎', '1',	125);
INSERT INTO poptbl2 VALUES('长崎', '2',	125);
INSERT INTO poptbl2 VALUES('东京', '1',	250);
INSERT INTO poptbl2 VALUES('东京', '2',	150);"""

# 8
sql = """
SELECT pref_name,
		SUM(CASE WHEN sex='1' THEN population ELSE 0 END) AS cnt_m,
		SUM(CASE WHEN sex='2' THEN population ELSE 0 END) AS cnt_f
	FROM poptbl2
	GROUP BY pref_name"""

# 14
sql = """
CREATE TABLE coursemaster
(course_id   INTEGER PRIMARY KEY,
course_name VARCHAR(32) NOT NULL);

INSERT INTO coursemaster VALUES(1, '会计入门');
INSERT INTO coursemaster VALUES(2, '财务知识');
INSERT INTO coursemaster VALUES(3, '簿记考试');
INSERT INTO coursemaster VALUES(4, '税务师');

CREATE TABLE opencourses
(month       INTEGER ,
course_id   INTEGER ,
PRIMARY KEY(month, course_id));

INSERT INTO opencourses VALUES(200706, 1);
INSERT INTO opencourses VALUES(200706, 3);
INSERT INTO opencourses VALUES(200706, 4);
INSERT INTO opencourses VALUES(200707, 4);
INSERT INTO opencourses VALUES(200708, 2);
INSERT INTO opencourses VALUES(200708, 4);"""

# 14
sql = """
SELECT course_name,
		CASE WHEN course_id in (SELECT course_id FROM opencourses WHERE month=200706) THEN 'O' ELSE 'X' END AS "6月",
		CASE WHEN course_id in (SELECT course_id FROM opencourses WHERE month=200707) THEN 'O' ELSE 'X' END AS "7月",
		CASE WHEN course_id in (SELECT course_id FROM opencourses WHERE month=200708) THEN 'O' ELSE 'X' END AS "8月"
	FROM coursemaster
		"""

# 22
sql = """
CREATE TABLE products
(name VARCHAR(16) PRIMARY KEY,
price INTEGER NOT NULL);

--可重排列·排列·组合
INSERT INTO products VALUES('苹果',	50);
INSERT INTO products VALUES('橘子',	100);
INSERT INTO products VALUES('香蕉',	80);"""

# 22
sql = """
SELECT p1.name AS name_1, p2.name AS name_2
	FROM products AS p1, products AS p2"""

# 23
sql = """
SELECT p1.name AS name_1, p2.name AS name_2
	FROM products AS p1, products AS p2
	WHERE p1.name <> p2.name"""

# 24
sql = """
SELECT p1.name AS name_1, p2.name AS name_2
	FROM products AS p1, products AS p2
	WHERE p1.name > p2.name"""

# 28
sql = """
CREATE TABLE addresses
(name VARCHAR(32),
family_id INTEGER,
address VARCHAR(32),
PRIMARY KEY(name, family_id));

INSERT INTO addresses VALUES('前田义明', '100', '东京都港区虎之门3-2-29');
INSERT INTO addresses VALUES('前田由美', '100', '东京都港区虎之门3-2-92');
INSERT INTO addresses VALUES('加藤茶',   '200', '东京都新宿区西新宿2-8-1');
INSERT INTO addresses VALUES('加藤胜',   '200', '东京都新宿区西新宿2-8-1');
INSERT INTO addresses VALUES('福尔摩斯',  '300', '贝克街221B');
INSERT INTO addresses VALUES('华生',  '400', '贝克街221B');"""

sql = """
SELECT a1.name, a1.address
	FROM addresses AS a1, addresses AS a2
	WHERE a1.family_id = a2.family_id
		AND a1.address <> a2.address"""


sql = """
DELETE FROM products;
INSERT INTO products VALUES('苹果',	50);
INSERT INTO products VALUES('橘子',	100);
INSERT INTO products VALUES('葡萄',	50);
INSERT INTO products VALUES('西瓜',	80);
INSERT INTO products VALUES('柠檬',	30);
INSERT INTO products VALUES('香蕉',	50);"""

sql = """
SELECT DISTINCT p1.name, p1.price
	FROM products AS p1, products AS P2
	WHERE p1.name <> p2.name
		AND p1.price = p2.price"""

# 30
sql = """
SELECT name, price,
		RANK() OVER (ORDER BY price DESC) AS rank_1,
		DENSE_RANK() OVER (ORDER BY price DESC) AS rank_2
	FROM products"""

sql = """
SELECT p1.name, p1.price,
		(SELECT COUNT(p2.price) FROM products AS p2 WHERE p2.price > p1.price) +1 AS rank_1
	FROM products AS p1
	ORDER BY rank_1"""

# 31
sql = """
SELECT p1.name, p1.price,
		(SELECT COUNT(DISTINCT p2.price) FROM products AS p2 WHERE p2.price > p1.price) +1 AS rank_1
	FROM products AS p1
	ORDER BY rank_1"""

sql = """
SELECT p1.name, p2.name
	FROM products AS p1 LEFT OUTER JOIN products AS p2
		ON p1.price < p2.price"""

# 59
sql = """
CREATE TABLE graduates
(name   VARCHAR(16) PRIMARY KEY,
income INTEGER NOT NULL);

INSERT INTO graduates VALUES('桑普森', 400000);
INSERT INTO graduates VALUES('迈克',     30000);
INSERT INTO graduates VALUES('怀特',   20000);
INSERT INTO graduates VALUES('阿诺德', 20000);
INSERT INTO graduates VALUES('史密斯',     20000);
INSERT INTO graduates VALUES('劳伦斯',   15000);
INSERT INTO graduates VALUES('哈德逊',   15000);
INSERT INTO graduates VALUES('肯特',     10000);
INSERT INTO graduates VALUES('贝克',   10000);
INSERT INTO graduates VALUES('斯科特',   10000);"""

sql = """
SELECT income, COUNT(*) AS cnt
	FROM graduates
	GROUP BY income
	HAVING COUNT(*) >= ALL(SELECT COUNT(*) FROM graduates GROUP BY income)"""

# 60
sql = """
SELECT income, COUNT(*) AS cnt
	FROM graduates
	GROUP BY income
	HAVING COUNT(*) >= (
		SELECT MAX(cnt)
			FROM (SELECT COUNT(*) AS cnt FROM graduates GROUP BY income) AS tmp)"""

sql = """
SELECT AVG(DISTINCT income)
	FROM
		(SELECT g1.income
				-- SUM(CASE WHEN g2.income >= g1.income THEN 1 ELSE 0 END) AS tmp_1,
				-- SUM(CASE WHEN g2.income <= g1.income THEN 1 ELSE 0 END) AS tmp_2,
				-- COUNT(*)/2 AS cnt_2
			FROM graduates AS g1, graduates AS g2
			GROUP BY g1.income
			HAVING SUM(CASE WHEN g2.income >= g1.income THEN 1 ELSE 0 END) >= COUNT(*)/2
				AND SUM(CASE WHEN g2.income <= g1.income THEN 1 ELSE 0 END) >= COUNT(*)/2) AS tmp"""

# 63
sql = """
CREATE TABLE students (
	student_id INTEGER NOT NULL,
	dpt VARCHAR(10) NOT NULL,
	sbmt_date DATE,
	PRIMARY KEY (student_id));
INSERT INTO students VALUES
	(100, '理学院', '2005-10-10'),
	(101, '理学院', '2005-09-22'),
	(102, '文学院', NULL),
	(103, '文学院', '2005-09-10'),
	(104, '文学院', '2005-09-22'),
	(105, '工学院',NULL),
	(106, '经济学院', '2005-09-25')"""

# 64
sql = """
SELECT dpt
	FROM students
	GROUP BY dpt
	HAVING COUNT(*) = COUNT(sbmt_date)"""

# 65
sql = """
CREATE TABLE Items
	(item VARCHAR(16) PRIMARY KEY);

CREATE TABLE ShopItems
	(shop VARCHAR(16),
	item VARCHAR(16),
	PRIMARY KEY(shop, item));

INSERT INTO Items VALUES('啤酒');
INSERT INTO Items VALUES('纸尿裤');
INSERT INTO Items VALUES('自行车');

INSERT INTO ShopItems VALUES('仙台',  '啤酒');
INSERT INTO ShopItems VALUES('仙台',  '纸尿裤');
INSERT INTO ShopItems VALUES('仙台',  '自行车');
INSERT INTO ShopItems VALUES('仙台',  '窗帘');
INSERT INTO ShopItems VALUES('东京',  '啤酒');
INSERT INTO ShopItems VALUES('东京',  '纸尿裤');
INSERT INTO ShopItems VALUES('东京',  '自行车');
INSERT INTO ShopItems VALUES('大阪',  '电视');
INSERT INTO ShopItems VALUES('大阪',  '纸尿裤');
INSERT INTO ShopItems VALUES('大阪',  '自行车');"""

# 67
sql = """
SELECT si.shop
	FROM items AS i, shopitems AS si
	WHERE si.item = i.item
	GROUP BY si.shop
	HAVING COUNT(si.item) = (SELECT COUNT(items) FROM items)"""

sql = """
SELECT si.shop
	FROM shopitems AS si LEFT OUTER JOIN items AS i
		ON si.item = i.item
	GROUP BY si.shop
	HAVING COUNT(si.item) = (SELECT COUNT(items) FROM items)
		AND COUNT(i.item) = (SELECT COUNT(items) FROM items)"""

# 73
sql = """
CREATE TABLE courses
(name   VARCHAR(32), 
course VARCHAR(32), 
PRIMARY KEY(name, course));

INSERT INTO courses VALUES('赤井', 'SQL入门');
INSERT INTO courses VALUES('赤井', 'UNIX基础');
INSERT INTO courses VALUES('铃木', 'SQL入门');
INSERT INTO courses VALUES('工藤', 'SQL入门');
INSERT INTO courses VALUES('工藤', 'Java中级');
INSERT INTO courses VALUES('吉田', 'UNIX基础');
INSERT INTO courses VALUES('渡边', 'SQL入门');"""

# 73
sql = """
SELECT c0.name,
		CASE WHEN c1.name IS NOT NULL THEN 'O' ELSE 'X' END AS "SQL入门",
		CASE WHEN c2.name IS NOT NULL THEN 'O' ELSE 'X' END AS "UNIX基础",
		CASE WHEN c3.name IS NOT NULL THEN 'O' ELSE 'X' END AS "Java中级"
	FROM (SELECT DISTINCT name FROM courses) AS c0
		LEFT OUTER JOIN (SELECT name FROM courses WHERE course = 'SQL入门') AS c1 ON c0.name = c1.name
		LEFT OUTER JOIN (SELECT name FROM courses WHERE course = 'UNIX基础') AS c2 ON c0.name = c2.name
		LEFT OUTER JOIN (SELECT name FROM courses WHERE course = 'Java中级') AS c3 ON c0.name = c3.name"""

# 75
sql = """
SELECT c0.name,
		(SELECT 'O' FROM courses AS c1 WHERE course='SQL入门' AND c1.name=c0.name) AS "SQL入门",
		(SELECT 'O' FROM courses AS c2 WHERE course='UNIX基础' AND c2.name=c0.name) AS "UNIX基础",
		(SELECT 'O' FROM courses AS c3 WHERE course='Java中级' AND c3.name=c0.name) AS "Java中级"
	FROM (SELECT DISTINCT name FROM courses) AS c0"""

sql = """
SELECT name,
		CASE WHEN SUM(CASE WHEN course = 'SQL入门' THEN 1 ELSE 0 END) = 1 THEN 'O' ELSE 'X' END AS "SQL入门",
		CASE WHEN SUM(CASE WHEN course = 'UNIX基础' THEN 1 ELSE 0 END) = 1 THEN 'O' ELSE 'X' END AS "UNIX基础",
		CASE WHEN SUM(CASE WHEN course = 'Java中级' THEN 1 ELSE 0 END) = 1 THEN 'O' ELSE 'X' END AS "Java中级"	
	FROM courses
	GROUP BY name"""

# 76
sql = """
CREATE TABLE Personnel
	(employee   varchar(32), 
	child_1    varchar(32), 
	child_2    varchar(32), 
	child_3    varchar(32), 
	PRIMARY KEY(employee));

INSERT INTO Personnel VALUES('赤井', '一郎', '二郎', '三郎');
INSERT INTO Personnel VALUES('工藤', '春子', '夏子', NULL);
INSERT INTO Personnel VALUES('铃木', '夏子', NULL,   NULL);
INSERT INTO Personnel VALUES('吉田', NULL,   NULL,   NULL);"""

# 77
sql = """
CREATE VIEW children(child) AS
SELECT child_1 FROM personnel
UNION
SELECT child_2 FROM personnel
UNION
SELECT child_3 FROM personnel"""

# 78
sql = """
SELECT e.employee, c.child, e.child_1, e.child_2, e.child_3
	FROM personnel AS e
		LEFT OUTER JOIN children AS c
			ON c.child IN (e.child_1, e.child_2, e.child_3)"""

# 79
sql = """
CREATE TABLE TblSex
(sex_cd   char(1), 
sex varchar(5), 
PRIMARY KEY(sex_cd));

CREATE TABLE TblAge 
(age_class char(1), 
age_range varchar(30), 
PRIMARY KEY(age_class));

CREATE TABLE TblPop 
(pref_name  varchar(30), 
age_class  char(1), 
sex_cd     char(1), 
population integer, 
PRIMARY KEY(pref_name, age_class,sex_cd));

INSERT INTO TblSex (sex_cd, sex ) VALUES('m',	'男');
INSERT INTO TblSex (sex_cd, sex ) VALUES('f',	'女');

INSERT INTO TblAge (age_class, age_range ) VALUES('1',	'21岁～30岁');
INSERT INTO TblAge (age_class, age_range ) VALUES('2',	'31岁～40岁');
INSERT INTO TblAge (age_class, age_range ) VALUES('3',	'41岁～50岁');

INSERT INTO TblPop VALUES('秋田', '1', 'm', 400 );
INSERT INTO TblPop VALUES('秋田', '3', 'm', 1000 );
INSERT INTO TblPop VALUES('秋田', '1', 'f', 800 );
INSERT INTO TblPop VALUES('秋田', '3', 'f', 1000 );
INSERT INTO TblPop VALUES('青森', '1', 'm', 700 );
INSERT INTO TblPop VALUES('青森', '1', 'f', 500 );
INSERT INTO TblPop VALUES('青森', '3', 'f', 800 );
INSERT INTO TblPop VALUES('东京', '1', 'm', 900 );
INSERT INTO TblPop VALUES('东京', '1', 'f', 1500 );
INSERT INTO TblPop VALUES('东京', '3', 'f', 1200 );
INSERT INTO TblPop VALUES('千叶', '1', 'm', 900 );
INSERT INTO TblPop VALUES('千叶', '1', 'f', 1000 );
INSERT INTO TblPop VALUES('千叶', '3', 'f', 900 );"""

# 81
sql = """
SELECT master.age_class, master.sex_cd, data.pop_tohokou, data.pop_kanto
	FROM 
		(SELECT tblage.age_class, tblsex.sex_cd
			FROM tblage CROSS JOIN tblsex) AS master LEFT OUTER JOIN
					(SELECT age_class, sex_cd,
							SUM(CASE WHEN pref_name in ('秋田', '青森') THEN population ELSE 0 END) AS pop_tohokou,
							SUM(CASE WHEN pref_name in ('东京', '千叶') THEN population ELSE 0 END) AS pop_kanto
						FROM tblpop
						GROUP BY age_class, sex_cd) AS data
				ON master.age_class = data.age_class AND master.sex_cd = data.sex_cd"""

# 85
sql = """
CREATE TABLE Class_A
(id char(1), 
name varchar(30), 
PRIMARY KEY(id));

CREATE TABLE Class_B
(id   char(1), 
name varchar(30), 
PRIMARY KEY(id));

INSERT INTO Class_A (id, name) VALUES('1', '田中');
INSERT INTO Class_A (id, name) VALUES('2', '铃木');
INSERT INTO Class_A (id, name) VALUES('3', '伊集院');

INSERT INTO Class_B (id, name) VALUES('1', '田中');
INSERT INTO Class_B (id, name) VALUES('2', '铃木');
INSERT INTO Class_B (id, name) VALUES('4', '西园寺');"""

# 86
sql = """
SELECT COALESCE(a.id, b.id), a.name, b.name
	FROM class_a AS a FULL OUTER JOIN class_b AS b
		ON a.id = b.id"""

# 88
sql = """
SELECT a.id, a.name
	FROM class_a AS a LEFT JOIN class_b AS b
		ON a.id = b.id
	WHERE b.name IS NULL"""

sql = """
SELECT *
	FROM items AS i LEFT OUTER JOIN shopitems AS si
		ON i.item = si.item"""

# 96
sql = """
CREATE TABLE Sales
(year INTEGER NOT NULL , 
sale INTEGER NOT NULL ,
PRIMARY KEY (year));

INSERT INTO Sales VALUES (1990, 50);
INSERT INTO Sales VALUES (1991, 51);
INSERT INTO Sales VALUES (1992, 52);
INSERT INTO Sales VALUES (1993, 52);
INSERT INTO Sales VALUES (1994, 50);
INSERT INTO Sales VALUES (1995, 50);
INSERT INTO Sales VALUES (1996, 49);
INSERT INTO Sales VALUES (1997, 55);"""

sql = """
SELECT year, sale
	FROM sales AS s1
	WHERE sale = (SELECT sale FROM sales AS s2 WHERE s2.year = s1.year-1)"""

sql = """
SELECT s1.year, s1.sale
	FROM sales AS s1, sales AS s2
	WHERE s1.sale = s2.sale
		AND s2.year = s1.year-1"""

# 97
sql = """
SELECT s1.year, s1.sale,
		CASE WHEN sale = (
				SELECT sale FROM sales AS s2 WHERE s2.year= s1.year-1) THEN '→'
			WHEN sale > (
				SELECT sale FROM sales AS s2 WHERE s2.year= s1.year-1) THEN '↑'
			WHEN sale < (
				SELECT sale FROM sales AS s2 WHERE s2.year= s1.year-1) THEN '↓'
			ELSE '-' END AS var			
	FROM sales AS s1"""

# 98
sql = """
SELECT s1.year, s1.sale,
		CASE WHEN s1.sale = s2.sale THEN '→'
			WHEN s1.sale > s2.sale THEN '↑'
			WHEN s1.sale < s2.sale THEN '↓'
			ELSE '-' END AS var
	FROM sales AS s1, sales AS s2
	WHERE s1.year = s2.year-1"""

# 99
sql = """
CREATE TABLE Sales2
(year INTEGER NOT NULL , 
sale INTEGER NOT NULL , 
PRIMARY KEY (year));

INSERT INTO Sales2 VALUES (1990, 50);
INSERT INTO Sales2 VALUES (1992, 50);
INSERT INTO Sales2 VALUES (1993, 52);
INSERT INTO Sales2 VALUES (1994, 55);
INSERT INTO Sales2 VALUES (1997, 55);"""

sql = """
SELECT year, sale
	FROM sales2 AS s1
	WHERE sale = (
		SELECT sale
			FROM sales2 AS s2
			WHERE s2.year = (
				SELECT MAX(year)
					FROM sales2 AS s3
					WHERE s1.year > s3.year))"""

# 100
sql = """
SELECT s2.year AS pre_year, s1.year AS now_year,
		s2.sale AS pre_sale, s1.sale AS now_sale,
		s1.sale - s2.sale AS diff
	FROM sales2 AS s1, sales2 AS s2
	-- 前一年等于所有小于当前年中的最大值
	WHERE s2.year = (SELECT MAX(year) FROM sales2 AS s3 WHERE s1.year > s3.year)"""

sql = """
SELECT s2.year AS pre_year, s1.year AS now_year,
		s2.sale AS pre_sale, s1.sale AS now_sale,
		s1.sale - s2.sale AS diff
	FROM sales2 AS s1 LEFT OUTER JOIN saleS2 AS s2
		ON s2.year = (SELECT MAX(year) FROM sales2 AS s3 WHERE s1.year>s3.year)"""

# 102
sql = """
CREATE TABLE Accounts
(prc_date DATE NOT NULL , 
prc_amt  INTEGER NOT NULL , 
PRIMARY KEY (prc_date)) ;

INSERT INTO Accounts VALUES ('2006-10-26',  12000 );
INSERT INTO Accounts VALUES ('2006-10-28',   2500 );
INSERT INTO Accounts VALUES ('2006-10-31', -15000 );
INSERT INTO Accounts VALUES ('2006-11-03',  34000 );
INSERT INTO Accounts VALUES ('2006-11-04',  -5000 );
INSERT INTO Accounts VALUES ('2006-11-06',   7200 );
INSERT INTO Accounts VALUES ('2006-11-11',  11000 );"""

sql = """
SELECT prc_date, prc_amt,
		SUM(prc_amt) OVER (ORDER BY prc_date),
		(SELECT SUM(prc_amt) FROM accounts AS a2 WHERE a1.prc_date>=a2.prc_date)
	FROM accounts AS a1"""

# 103
sql = """
SELECT prc_date, prc_amt,
		SUM(prc_amt) OVER (ORDER BY prc_date ROWS 2 PRECEDING),
		(
			SELECT SUM(prc_amt)
				FROM accounts AS a2
				WHERE a1.prc_date >= a2.prc_date
					AND (
						SELECT COUNT(*)
							FROM accounts AS a3
							WHERE a3.prc_date BETWEEN a2.prc_date AND a1.prc_date) <=3)
	FROM accounts AS a1"""

# 106
sql = """
CREATE TABLE reservations
(reserver VARCHAR(30) PRIMARY KEY,
start_date  DATE  NOT NULL,
end_date    DATE  NOT NULL);

INSERT INTO reservations VALUES('木村', '2006-10-26', '2006-10-27');
INSERT INTO reservations VALUES('荒木', '2006-10-28', '2006-10-31');
INSERT INTO reservations VALUES('堀',   '2006-10-31', '2006-11-01');
INSERT INTO reservations VALUES('山本', '2006-11-03', '2006-11-04');
INSERT INTO reservations VALUES('内田', '2006-11-03', '2006-11-05');
INSERT INTO reservations VALUES('水谷', '2006-11-06', '2006-11-06');"""

# 106
sql = """
SELECT reserver, start_date, end_date
	FROM reservations AS r1
	WHERE EXISTS (
		SELECT *
			FROM reservations AS r2
			WHERE r1.reserver <> r2.reserver
				AND (r1.start_date BETWEEN r2.start_date AND r2.end_date
					OR r1.end_date BETWEEN r2.start_date AND r2.end_date))"""

# 120
sql = """
CREATE TABLE Skills 
(skill VARCHAR(32),
PRIMARY KEY(skill));

CREATE TABLE EmpSkills 
(emp   VARCHAR(32), 
skill VARCHAR(32),
PRIMARY KEY(emp, skill));

INSERT INTO Skills VALUES('Oracle');
INSERT INTO Skills VALUES('UNIX');
INSERT INTO Skills VALUES('Java');

INSERT INTO EmpSkills VALUES('相田', 'Oracle');
INSERT INTO EmpSkills VALUES('相田', 'UNIX');
INSERT INTO EmpSkills VALUES('相田', 'Java');
INSERT INTO EmpSkills VALUES('相田', 'C#');
INSERT INTO EmpSkills VALUES('神崎', 'Oracle');
INSERT INTO EmpSkills VALUES('神崎', 'UNIX');
INSERT INTO EmpSkills VALUES('神崎', 'Java');
INSERT INTO EmpSkills VALUES('平井', 'UNIX');
INSERT INTO EmpSkills VALUES('平井', 'Oracle');
INSERT INTO EmpSkills VALUES('平井', 'PHP');
INSERT INTO EmpSkills VALUES('平井', 'Perl');
INSERT INTO EmpSkills VALUES('平井', 'C++');
INSERT INTO EmpSkills VALUES('若田部', 'Perl');
INSERT INTO EmpSkills VALUES('渡来', 'Oracle');"""

sql = """
SELECT DISTINCT emp
	FROM empskills AS es1
	WHERE NOT EXISTS (
		SELECT * FROM skills
			EXCEPT
		SELECT skill FROM empskills AS es2 WHERE es2.emp=es1.emp)"""

# 122
sql = """
CREATE TABLE supparts
(sup  CHAR(32) NOT NULL,
part CHAR(32) NOT NULL,
PRIMARY KEY(sup, part));

INSERT INTO supparts VALUES('A',  '螺丝');
INSERT INTO supparts VALUES('A',  '螺母');
INSERT INTO supparts VALUES('A',  '管子');
INSERT INTO supparts VALUES('B',  '螺丝');
INSERT INTO supparts VALUES('B',  '管子');
INSERT INTO supparts VALUES('C',  '螺丝');
INSERT INTO supparts VALUES('C',  '螺母');
INSERT INTO supparts VALUES('C',  '管子');
INSERT INTO supparts VALUES('D',  '螺丝');
INSERT INTO supparts VALUES('D',  '管子');
INSERT INTO supparts VALUES('E',  '保险丝');
INSERT INTO supparts VALUES('E',  '螺母');
INSERT INTO supparts VALUES('E',  '管子');
INSERT INTO supparts VALUES('F',  '保险丝');"""

# 124
sql = """
SELECT s1.sup AS s1, s2.sup AS s2
	FROM supparts AS s1, supparts AS s2
	WHERE s1.sup < s2.sup	--自联结供应商组合
		AND s1.part = s2.part --条件1，零件相同
	GROUP BY s1.sup, s2.sup
	HAVING COUNT(*) = (SELECT COUNT(*) FROM supparts AS s3 WHERE s3.sup = s1.sup)	--条件2，零件数目相同，防止两个合集被包含的情况
		AND COUNT(*) = (SELECT COUNT(*) FROM supparts AS s4 WHERE s4.sup = s2.sup)"""

# 136
sql = """
CREATE TABLE Meetings
(meeting CHAR(32) NOT NULL,
person  CHAR(32) NOT NULL,
PRIMARY KEY (meeting, person));

INSERT INTO Meetings VALUES('第1次', '伊藤');
INSERT INTO Meetings VALUES('第1次', '水岛');
INSERT INTO Meetings VALUES('第1次', '坂东');
INSERT INTO Meetings VALUES('第2次', '伊藤');
INSERT INTO Meetings VALUES('第2次', '宫田');
INSERT INTO Meetings VALUES('第3次', '坂东');
INSERT INTO Meetings VALUES('第3次', '水岛');
INSERT INTO Meetings VALUES('第3次', '宫田');
"""

sql = """
SELECT DISTINCT m1.meeting, m2.person
	-- FROM meetings AS m1 CROSS JOIN meetings AS m2
	FROM meetings AS m1, meetings AS m2
	WHERE NOT EXISTS(
		SELECT *
			FROM meetings AS m3
			WHERE m3.meeting = m1.meeting
				AND m3.person = m2.person)"""

# 138
sql = """
CREATE TABLE testscores
(student_id INTEGER,
subject    VARCHAR(32) ,
score      INTEGER,
PRIMARY KEY(student_id, subject));

INSERT INTO testscores VALUES(100, '数学',100);
INSERT INTO testscores VALUES(100, '语文',80);
INSERT INTO testscores VALUES(100, '理化',80);
INSERT INTO testscores VALUES(200, '数学',80);
INSERT INTO testscores VALUES(200, '语文',95);
INSERT INTO testscores VALUES(300, '数学',40);
INSERT INTO testscores VALUES(300, '语文',90);
INSERT INTO testscores VALUES(300, '社会',55);
INSERT INTO testscores VALUES(400, '数学',80);"""

sql = """
SELECT DISTINCT student_id
	FROM testscores AS ts1
	WHERE NOT EXISTS(
		SELECT *
			FROM testscores AS ts2
			WHERE ts2.student_id = ts1.student_id
				AND ts2.score < 50)"""

# 140
sql = """
SELECT student_id
	FROM testscores AS ts1
	WHERE subject in ('语文', '数学')
		AND NOT EXISTS(
			SELECT *
				FROM testscores AS ts2
				WHERE ts2.student_id = ts1.student_id
					AND 1 = (
						CASE WHEN subject = '语文' AND score < 50 THEN 1
							WHEN subject = '数学' AND score < 80 THEN 1
							ELSE 0 END))
	GROUP BY student_id
	HAVING COUNT(*)=2"""

# 141
sql = """
CREATE TABLE Projects
(project_id VARCHAR(32),
step_nbr   INTEGER ,
status     VARCHAR(32),
PRIMARY KEY(project_id, step_nbr));

INSERT INTO Projects VALUES('AA100', 0, '完成');
INSERT INTO Projects VALUES('AA100', 1, '等待');
INSERT INTO Projects VALUES('AA100', 2, '等待');
INSERT INTO Projects VALUES('B200',  0, '等待');
INSERT INTO Projects VALUES('B200',  1, '等待');
INSERT INTO Projects VALUES('CS300', 0, '完成');
INSERT INTO Projects VALUES('CS300', 1, '完成');
INSERT INTO Projects VALUES('CS300', 2, '等待');
INSERT INTO Projects VALUES('CS300', 3, '等待');
INSERT INTO Projects VALUES('DY400', 0, '完成');
INSERT INTO Projects VALUES('DY400', 1, '完成');
INSERT INTO Projects VALUES('DY400', 2, '完成');"""

sql = """
SELECT project_id
	FROM projects
	GROUP BY project_id
	HAVING COUNT(*) = SUM(
		CASE WHEN step_nbr<=1 AND status='完成' THEN 1
			WHEN step_nbr>1 AND status='等待' THEN 1
			ELSE 0 END)"""

sql = """
SELECT *
	FROM projects AS p1
	WHERE NOT EXISTS (
		SELECT *
			FROM projects AS p2
			WHERE p2.project_id = p1.project_id
				AND status <>(
					CASE WHEN step_nbr <= 1 THEN '完成'
						ELSE '等待' END))"""

# 156
sql = """
CREATE TABLE seats
(seat   INTEGER NOT NULL  PRIMARY KEY,
status CHAR(6) NOT NULL
CHECK (status IN ('未预订', '已预订')) ); 

INSERT INTO seats VALUES (1,  '已预订');
INSERT INTO seats VALUES (2,  '已预订');
INSERT INTO seats VALUES (3,  '未预订');
INSERT INTO seats VALUES (4,  '未预订');
INSERT INTO seats VALUES (5,  '未预订');
INSERT INTO seats VALUES (6,  '已预订');
INSERT INTO seats VALUES (7,  '未预订');
INSERT INTO seats VALUES (8,  '未预订');
INSERT INTO seats VALUES (9,  '未预订');
INSERT INTO seats VALUES (10,  '未预订');
INSERT INTO seats VALUES (11,  '未预订');
INSERT INTO seats VALUES (12,  '已预订');
INSERT INTO seats VALUES (13,  '已预订');
INSERT INTO seats VALUES (14,  '未预订');
INSERT INTO seats VALUES (15,  '未预订');"""

sql = """
SELECT s1.seat AS start_seat, '~', s2.seat AS end_seat
	FROM seats AS s1, seats AS s2
	WHERE s2.seat = s1.seat+2
		AND NOT EXISTS(
			SELECT *
				FROM seats AS s3
				WHERE s3.seat BETWEEN s1.seat AND s2.seat
					AND s3.status <> '未预订')"""

# 158
sql = """
CREATE TABLE Seats2
( seat   INTEGER NOT NULL  PRIMARY KEY,
row_id CHAR(1) NOT NULL,
status CHAR(6) NOT NULL
CHECK (status IN ('未预订', '已预订')) ); 

INSERT INTO Seats2 VALUES (1, 'A', '已预订');
INSERT INTO Seats2 VALUES (2, 'A', '已预订');
INSERT INTO Seats2 VALUES (3, 'A', '未预订');
INSERT INTO Seats2 VALUES (4, 'A', '未预订');
INSERT INTO Seats2 VALUES (5, 'A', '未预订');
INSERT INTO Seats2 VALUES (6, 'B', '已预订');
INSERT INTO Seats2 VALUES (7, 'B', '已预订');
INSERT INTO Seats2 VALUES (8, 'B', '未预订');
INSERT INTO Seats2 VALUES (9, 'B', '未预订');
INSERT INTO Seats2 VALUES (10,'B', '未预订');
INSERT INTO Seats2 VALUES (11,'C', '未预订');
INSERT INTO Seats2 VALUES (12,'C', '未预订');
INSERT INTO Seats2 VALUES (13,'C', '未预订');
INSERT INTO Seats2 VALUES (14,'C', '已预订');
INSERT INTO Seats2 VALUES (15,'C', '未预订');"""

sql = """
SELECT s1.seat AS start_seat, '~', s2.seat AS end_seat
	FROM seats2 AS s1, seats2 AS s2
	WHERE s2.seat = s1.seat+2
		AND NOT EXISTS(
			SELECT *
				FROM seats2 AS s3
				WHERE s3.seat BETWEEN s1.seat AND s2.seat
					AND (s3.status <> '未预订'
						OR s3.row_id <> s1.row_id))
-- not (s3.status = '未预订' and s3.row_id = s1.row_id)
-- = (s3.status <> '未预订' OR s3.row_id <> s1.row_id)"""

# 159
sql = """
CREATE TABLE Seats3
( seat   INTEGER NOT NULL  PRIMARY KEY,
status CHAR(6) NOT NULL
CHECK (status IN ('未预订', '已预订')) ); 

INSERT INTO Seats3 VALUES (1,  '已预订');
INSERT INTO Seats3 VALUES (2,  '未预订');
INSERT INTO Seats3 VALUES (3,  '未预订');
INSERT INTO Seats3 VALUES (4,  '未预订');
INSERT INTO Seats3 VALUES (5,  '未预订');
INSERT INTO Seats3 VALUES (6,  '已预订');
INSERT INTO Seats3 VALUES (7,  '未预订');
INSERT INTO Seats3 VALUES (8,  '已预订');
INSERT INTO Seats3 VALUES (9,  '未预订');
INSERT INTO Seats3 VALUES (10, '未预订');
"""

# 156
sql = """
CREATE VIEW sequence (start_seat, end_seat, seat_cnt) AS
SELECT s1.seat, s2.seat, s2.seat-s1.seat+1 AS seat_cnt
	FROM seats3 AS s1, seats3 AS s2
	WHERE s1.seat < s2.seat
		AND NOT EXISTS (
			SELECT *
				FROM seats3 AS s3
				WHERE (s3.seat BETWEEN  s1.seat AND s2.seat AND s3.status<>'未预订')
					OR (s3.seat = s1.seat-1 AND s3.status = '未预订')
					OR (s3.seat = s2.seat+1 AND s3.status = '未预订'));
	--同上，双重否定转化
	
SELECT *
	FROM sequence
	WHERE seat_cnt = (
		SELECT MAX(seat_cnt)
			FROM sequence)"""

# 163
sql = """
CREATE TABLE MyStock
(deal_date  DATE PRIMARY KEY,
price      INTEGER ); 

INSERT INTO MyStock VALUES ('2007-01-06', 1000);
INSERT INTO MyStock VALUES ('2007-01-08', 1050);
INSERT INTO MyStock VALUES ('2007-01-09', 1050);
INSERT INTO MyStock VALUES ('2007-01-12', 900);
INSERT INTO MyStock VALUES ('2007-01-13', 880);
INSERT INTO MyStock VALUES ('2007-01-14', 870);
INSERT INTO MyStock VALUES ('2007-01-16', 920);
INSERT INTO MyStock VALUES ('2007-01-17', 1000);"""

sql = """
SELECT ms1.deal_date AS start_date, ms2.deal_date AS end_date
	FROM mystock AS ms1, mystock AS ms2
	WHERE ms1.deal_date < ms2.deal_date
		AND NOT EXISTS (
			SELECT *
				FROM mystock AS ms3, mystock AS ms4
				WHERE ms3.deal_date < ms4.deal_date
					AND ms3.deal_date BETWEEN ms1.deal_date AND ms2.deal_date
					AND ms4.deal_date BETWEEN ms1.deal_date AND ms2.deal_date
					AND ms3.price >= ms4.price)"""

# 168
sql = """
CREATE TABLE teams
(member  CHAR(12) NOT NULL PRIMARY KEY,
team_id INTEGER  NOT NULL,
status  CHAR(8)  NOT NULL);

INSERT INTO teams VALUES('乔',     1, '待命');
INSERT INTO teams VALUES('肯',     1, '出勤中');
INSERT INTO teams VALUES('米克',   1, '待命');
INSERT INTO teams VALUES('卡伦',   2, '出勤中');
INSERT INTO teams VALUES('凯斯',   2, '休息');
INSERT INTO teams VALUES('简',     3, '待命');
INSERT INTO teams VALUES('哈特',   3, '待命');
INSERT INTO teams VALUES('迪克',   3, '待命');
INSERT INTO teams VALUES('贝斯',   4, '待命');
INSERT INTO teams VALUES('阿伦',   5, '出勤中');
INSERT INTO teams VALUES('罗伯特', 5, '休息');
INSERT INTO teams VALUES('卡根',   5, '待命');"""

sql = """
SELECT team_id, member
	FROM teams AS t1
	WHERE NOT EXISTS (
		SELECT *
			FROM teams AS t2
			WHERE t2.team_id = t1.team_id
				AND t2.status <> '待命')"""

sql = """
SELECT team_id
	FROM teams
	GROUP BY team_id
	-- HAVING COUNT(*) = SUM(CASE WHEN status = '待命' THEN 1 ELSE 0 END)
	HAVING MIN(status) = '待命' AND MAX(status) = '待命'"""

# 172
sql = """
CREATE TABLE Materials
(center         CHAR(12) NOT NULL,
receive_date   DATE     NOT NULL,
material       CHAR(12) NOT NULL,
PRIMARY KEY(center, receive_date));

INSERT INTO Materials VALUES('东京'	,'2007-4-01',	'锡');
INSERT INTO Materials VALUES('东京'	,'2007-4-12',	'锌');
INSERT INTO Materials VALUES('东京'	,'2007-5-17',	'铝');
INSERT INTO Materials VALUES('东京'	,'2007-5-20',	'锌');
INSERT INTO Materials VALUES('大阪'	,'2007-4-20',	'铜');
INSERT INTO Materials VALUES('大阪'	,'2007-4-22',	'镍');
INSERT INTO Materials VALUES('大阪'	,'2007-4-29',	'铅');
INSERT INTO Materials VALUES('名古屋',	'2007-3-15',	'钛');
INSERT INTO Materials VALUES('名古屋',	'2007-4-01',	'钢');
INSERT INTO Materials VALUES('名古屋',	'2007-4-24',	'钢');
INSERT INTO Materials VALUES('名古屋',	'2007-5-02',	'镁');
INSERT INTO Materials VALUES('名古屋',	'2007-5-10',	'钛');
INSERT INTO Materials VALUES('福冈'	,'2007-5-10',	'锌');
INSERT INTO Materials VALUES('福冈'	,'2007-5-28',	'锡');"""

sql = """
SELECT center
	FROM materials
	GROUP BY center
	HAVING COUNT(material) <> COUNT(DISTINCT material)"""

sql = """
SELECT center, material
	FROM materials AS m1
	WHERE EXISTS(
		SELECT *
			FROM materials AS m2
			WHERE m1.center = m2.center
				AND m1.receive_date <> m2.receive_date
				AND m1.material = m2.material)"""

# 176
sql = """
SELECT CASE WHEN COUNT(*)=1 OR MIN(seq) > 1 THEN 1
		ELSE (
			SELECT MIN(seq+1)
				FROM seqtbl AS s1
				WHERE NOT EXISTS (
					SELECT *
						FROM seqtbl AS s2
						WHERE s2.seq = s1.seq+1)
		"""

# 178
sql = """

CREATE TABLE testresults
(student CHAR(12) NOT NULL PRIMARY KEY,
class   CHAR(1)  NOT NULL,
sex     CHAR(1)  NOT NULL,
score   INTEGER  NOT NULL);

INSERT INTO testresults VALUES('001', 'A', '男', 100);
INSERT INTO testresults VALUES('002', 'A', '女', 100);
INSERT INTO testresults VALUES('003', 'A', '女',  49);
INSERT INTO testresults VALUES('004', 'A', '男',  30);
INSERT INTO testresults VALUES('005', 'B', '女', 100);
INSERT INTO testresults VALUES('006', 'B', '男',  92);
INSERT INTO testresults VALUES('007', 'B', '男',  80);
INSERT INTO testresults VALUES('008', 'B', '男',  80);
INSERT INTO testresults VALUES('009', 'B', '女',  10);
INSERT INTO testresults VALUES('010', 'C', '男',  92);
INSERT INTO testresults VALUES('011', 'C', '男',  80);
INSERT INTO testresults VALUES('012', 'C', '女',  21);
INSERT INTO testresults VALUES('013', 'D', '女', 100);
INSERT INTO testresults VALUES('014', 'D', '女',   0);
INSERT INTO testresults VALUES('015', 'D', '女',   0);
"""

# 178
sql = """
SELECT class
	FROM testresults
	GROUP BY class
	HAVING COUNT(*)*0.75 <= SUM(
		CASE WHEN score >=80 THEN 1 ELSE 0 END)"""

sql = """
SELECT class
	FROM testresults
	GROUP BY class
	HAVING SUM(CASE WHEN score>=50 and sex='男' THEN 1 ELSE 0 END)>
		SUM(CASE WHEN score>=50 and sex='女' THEN 1 ELSE 0 END)"""

sql = """
SELECT class
	FROM testresults
	GROUP BY class
	HAVING AVG(CASE WHEN sex='男' THEN score ELSE NULL END) <
		AVG(CASE WHEN sex='女' THEN score ELSE NULL END)"""











df = pd.read_sql(sql, conn)
print(df)

cursor.execute(sql)
conn.commit()
cursor.close()
conn.close()