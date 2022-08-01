-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema brats
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema brats
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `brats` DEFAULT CHARACTER SET utf8mb4 ;
USE `brats` ;

-- -----------------------------------------------------
-- Table `brats`.`tb_patients`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `brats`.`tb_patients` (
  `patient_id` VARCHAR(20) NOT NULL,
  UNIQUE INDEX `patient_name_UNIQUE` (`patient_id` ASC) VISIBLE,
  PRIMARY KEY (`patient_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `brats`.`tb_modalities`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `brats`.`tb_modalities` (
  `stamp` VARCHAR(60) NOT NULL,
  `patient_id` VARCHAR(20) NOT NULL,
  `train_or_validate` VARCHAR(10) NOT NULL,
  `modality` VARCHAR(100) NOT NULL,
  `remark` VARCHAR(100) NOT NULL,
  `is_basic` TINYINT NOT NULL,
  `path` VARCHAR(300) NOT NULL,
  `meta_data` VARCHAR(1000) NOT NULL,
  INDEX `pat_name_idx` (`patient_id` ASC) INVISIBLE,
  UNIQUE INDEX `mod_stamp_UNIQUE` (`stamp` ASC) VISIBLE,
  PRIMARY KEY (`stamp`),
  CONSTRAINT `name`
    FOREIGN KEY (`patient_id`)
    REFERENCES `brats`.`tb_patients` (`patient_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `brats`.`tb_patches`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `brats`.`tb_patches` (
  `patch_stamp` VARCHAR(100) NOT NULL,
  `original_stamp` VARCHAR(60) NOT NULL,
  `patch_path` VARCHAR(300) NOT NULL,
  `patch_meta_data` VARCHAR(1000) NOT NULL,
  PRIMARY KEY (`patch_stamp`),
  UNIQUE INDEX `patch_stamp_UNIQUE` (`patch_stamp` ASC) VISIBLE,
  INDEX `patch's original stamp_idx` (`original_stamp` ASC) VISIBLE,
  CONSTRAINT `patch's original stamp`
    FOREIGN KEY (`original_stamp`)
    REFERENCES `brats`.`tb_modalities` (`stamp`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
