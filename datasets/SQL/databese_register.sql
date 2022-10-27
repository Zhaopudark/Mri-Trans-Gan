-- Active: 1666134863355@@127.0.0.1@3306@brats
drop user if exists 'guest'@'localhost';
-- create user 'guest'@'localhost' identified by 'wR53T-jPRgZ!62F';
create user 'guest'@'localhost' identified by '123456';
grant create, select, insert, delete, update  on *.* to 'guest'@'localhost';


drop user if exists 'guest_remote';
create user 'guest_remote' identified by '123456';
grant create, select, insert, delete, update,DROP on *.* to 'guest_remote';
flush privileges;


-- grant all privileges on hrs.* to 'guest'@'localhost';pat_id
-- grant all privileges on *.* to 'guest'@'localhost' with grant option;
-- revoke insert on hrs.*  from  'guest'@'localhost';