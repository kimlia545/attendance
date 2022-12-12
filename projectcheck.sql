-- --------------------------------------------------------
-- 호스트:                          127.0.0.1
-- 서버 버전:                        10.5.4-MariaDB - mariadb.org binary distribution
-- 서버 OS:                        Win64
-- HeidiSQL 버전:                  11.0.0.5919
-- --------------------------------------------------------

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET NAMES utf8 */;
/*!50503 SET NAMES utf8mb4 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;


-- abscheck 데이터베이스 구조 내보내기
CREATE DATABASE IF NOT EXISTS `abscheck` /*!40100 DEFAULT CHARACTER SET utf8mb4 */;
USE `abscheck`;

-- 테이블 abscheck.abs_log 구조 내보내기
CREATE TABLE IF NOT EXISTS `abs_log` (
  `idx` int(11) NOT NULL AUTO_INCREMENT,
  `m_id` int(11) DEFAULT 0,
  `isin` tinyint(1) DEFAULT NULL COMMENT '0출석, 1퇴실',
  `regdate` datetime DEFAULT NULL,
  PRIMARY KEY (`idx`)
) ENGINE=InnoDB AUTO_INCREMENT=61 DEFAULT CHARSET=utf8;

-- 내보낼 데이터가 선택되어 있지 않습니다.

-- 테이블 abscheck.admin 구조 내보내기
CREATE TABLE IF NOT EXISTS `admin` (
  `uid` varchar(50) NOT NULL,
  `upw` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- 내보낼 데이터가 선택되어 있지 않습니다.

-- 테이블 abscheck.member_list 구조 내보내기
CREATE TABLE IF NOT EXISTS `member_list` (
  `no` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `name` varchar(45) CHARACTER SET utf8mb4 NOT NULL,
  `class` varchar(45) CHARACTER SET utf8mb4 NOT NULL DEFAULT '인공지능',
  `birdate` date DEFAULT NULL,
  `gender` char(2) CHARACTER SET utf8mb4 NOT NULL,
  `abs_rate` tinytext DEFAULT 'X' COMMENT '출석여부',
  PRIMARY KEY (`no`)
) ENGINE=InnoDB AUTO_INCREMENT=16 DEFAULT CHARSET=utf8;

-- 내보낼 데이터가 선택되어 있지 않습니다.

/*!40101 SET SQL_MODE=IFNULL(@OLD_SQL_MODE, '') */;
/*!40014 SET FOREIGN_KEY_CHECKS=IF(@OLD_FOREIGN_KEY_CHECKS IS NULL, 1, @OLD_FOREIGN_KEY_CHECKS) */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
