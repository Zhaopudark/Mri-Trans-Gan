drop user if exists 'guest'@'localhost'  ;
create user 'guest'@'localhost' identified by '123456';
grant create, select, insert, delete, update  on *.* to 'guest'@'localhost';
-- grant all privileges on hrs.* to 'guest'@'localhost';pat_id
-- grant all privileges on *.* to 'guest'@'localhost' with grant option;
-- revoke insert on hrs.*  from  'guest'@'localhost';