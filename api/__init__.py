import os
from dotenv import load_dotenv
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

load_dotenv()
