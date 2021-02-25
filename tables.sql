CREATE TABLE IF NOT EXISTS `faq_management_model`(
   `record_id` int(10) primary key auto_increment,
   `name` VARCHAR(100) NOT NULL,
   `domain` VARCHAR(100) NOT NULL,
   `create_time` DATETIME DEFAULT CURRENT_TIMESTAMP,
   `state` VARCHAR(100) NOT NULL,
   `categories` VARCHAR(100) NOT NULL,
   `category_num` int(10) NOT NULL,
   `comment` VARCHAR(100) NOT NULL
)ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `faq_management_category`(
   `category_id` int(10) primary key auto_increment,
   `name` VARCHAR(100) NOT NULL,
   `answer` TEXT NOT NULL,
   `create_time` DATETIME DEFAULT CURRENT_TIMESTAMP
)ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `faq_management_query`(
   `query_id` int(10) primary key auto_increment,
   `category_id` VARCHAR(100) NOT NULL,
   `text` TEXT NOT NULL,
   `create_time` DATETIME DEFAULT CURRENT_TIMESTAMP
)ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
