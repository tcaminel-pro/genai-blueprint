from datetime import datetime, timedelta
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

TABLE_NAME = "maintenance_planning"
DATA_PATH = Path.cwd() / "use_case_data/maintenance"
MODEL_CACHE = Path.cwd() / ".model_cache"

assert DATA_PATH.exists()


@cache
def dummy_database() -> str:
    """Create a dummy database  for the demo. Returns the SqlAlchemy URI.

    We create:
    - a table with sensor events
    - a table with employees planning in the last few weeks
    """
    DATABASE_URI = f"sqlite:///{DATA_PATH}/demo.db"
    # fmt: off
    proc = "Power Plant Steam Turbine"
    iot_warnings = [
        {"timestamp": "2023-01-01 10:00:00", "sensor": "sensor1", "warning": "Event 1"},
        {"timestamp": "2023-01-02 14:30:00", "sensor": "sensor1", "warning": "Event 2"},
        {"timestamp": "2023-01-03 09:15:00", "sensor": "sensor2", "warning": "Event 3"},
        {"timestamp": "2023-09-01 10:00:00", "sensor": "sensor2", "warning": "Event 4"},
        {"timestamp": "2023-09-02 14:30:00", "sensor": "sensor1", "warning": "Event 5"},
        {"timestamp": "2023-09-03 09:15:00", "sensor": "sensor2", "warning": "Event 6"},
        {"timestamp": "2023-10-01 10:00:00", "sensor": "sensor2", "warning": "Event 7"},
        {"timestamp": "2023-10-02 14:30:00", "sensor": "sensor1", "warning": "Event 8"},
        {"timestamp": "2023-10-03 09:15:00", "sensor": "sensor2", "warning": "Event 9"},
    ]
    def today_minus(n: int) -> datetime:
        return datetime.today() - timedelta(days=n)
    def today_plus(n: int) -> datetime:
        return datetime.today() + timedelta(days=n)

    tasks = [
        {"employee": "John Smith", "start_date": today_plus(0), "end_date": today_plus(1), "procedure": proc, "task": "Preparations for Maintenance"},
        {"employee": "John Smith", "start_date": today_plus(0), "end_date": today_plus(3), "procedure": proc, "task": "Turbine Shutdown"},
        {"employee": "John Smith", "start_date": today_plus(3),  "end_date": today_plus(5), "procedure": proc, "task": "Rotor Inspection"},
        {"employee": "John Smith", "start_date": today_plus(5),  "end_date": today_plus(12), "procedure": proc, "task": "Blade Inspection"},
        {"employee": "John Smith", "start_date": today_plus(6),  "end_date": today_plus(15), "procedure": proc, "task": "Diaphragm Inspection"},
        {"employee": "John Smith", "start_date": today_plus(8),  "end_date": today_plus(13), "procedure": proc, "task": "Bearing Inspection"},
        {"employee": "John Smith", "start_date": today_plus(10),  "end_date": today_plus(12), "procedure": proc, "task": "Final Checks and Cleanup"},
        {"employee": "John Smith", "start_date": today_plus(12), "end_date": today_plus(14), "procedure": proc, "task": "Preparations for Maintenance"},
        
        {"employee": "Alice Johnson", "start_date": today_plus(0), "end_date": today_plus(2), "procedure": proc, "task": "Turbine Shutdown"},
        {"employee": "Alice Johnson", "start_date": today_plus(2),  "end_date": today_plus(4), "procedure": proc, "task": "Rotor Inspection"},
        {"employee": "Alice Johnson", "start_date": today_plus(4),  "end_date": today_plus(7), "procedure": proc, "task": "Blade Inspection"},
        {"employee": "Alice Johnson", "start_date": today_plus(6),  "end_date": today_plus(12), "procedure": proc, "task": "Diaphragm Inspection"},
        {"employee": "Alice Johnson", "start_date": today_plus(8),  "end_date": today_plus(9), "procedure": proc, "task": "Bearing Inspection"},
        {"employee": "Alice Johnson", "start_date": today_plus(10),  "end_date": today_plus(14), "procedure": proc, "task": "Final Checks and Cleanup"},                  
    ]

    sensors = [
        ("signal_1", "Volt", 30), 
        ("signal_2", "A", 60), 
        ("signal_3", "SI", 90), 
    ]
    dates = pd.date_range(start=today_minus(90), end=today_minus(0))
    sensor_data = pd.DataFrame(columns=['date', 'sensor', 'value', 'unit'])
    for name, unit, period in sensors:
        sensor_values = np.sin(2 * np.pi * (dates.dayofyear / period))
        sensor_df = pd.DataFrame({'date': dates, 'sensor': name, 'value': sensor_values, 'unit': unit})
        sensor_data = pd.concat([sensor_data, sensor_df], ignore_index=True)

    # fmt: on
    logger.info(f"create database {DATABASE_URI}")
    for name, tables in [("iot_warnings", iot_warnings), ("tasks", tasks)]:
        df = pd.DataFrame(tables)
        df.to_sql(name, DATABASE_URI, if_exists="replace", index=False)

    sensor_data.to_sql("sensor_data", DATABASE_URI, if_exists="replace", index=False)
    return DATABASE_URI
