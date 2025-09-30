def check_dsn_update_driver(db_url: str, driver: str | None = None) -> str:
    """Check a Database Source Name (DSN) compliant with SQLAlchemy URL format.
    The driver part of the connection can be changed (ex: postgress+"asyncpg")"""
    from sqlalchemy.engine.url import make_url

    try:
        url = make_url(db_url)
        # If driver is specified and not already in the URL, add it
        if driver and "+" not in str(url.drivername):
            drivername = f"{url.drivername}+{driver}"
            new_url = url.set(drivername=drivername)
            return new_url.render_as_string(hide_password=False)

        return url.render_as_string(hide_password=False)

        #     self.engine = create_engine(dsn)
        #     # Test connection
        #     with self.engine.connect() as conn:
        #         conn.execute(text("SELECT 1"))
        # except Exception as e:
        #     raise ConnectionError(f"Failed to connect to database using DSN'{self.dsn}': {e}") from e

    except Exception as ex:
        raise ValueError(f"Incorrect database DSN: {db_url} (driver: {driver or 'unchanged'}).") from ex
