# scripts/init_db_from_snapshot.py

import os
import time
import subprocess
from dotenv import load_dotenv

def main():
    load_dotenv()

    # db_host = os.getenv("DB_HOST", "localhost")
    # db_port = os.getenv("DB_PORT", "5432")
    # db_user = os.getenv("DB_USER", "user")
    # db_password = os.getenv("DB_PASSWORD", "password")
    # db_name = os.getenv("DB_NAME", "database")

    snapshot_path = os.path.join("data", "fraudData", "fraudData_snapshot.dump")
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(
            f"Snapshot file not found at {snapshot_path}. "
        )

    if subprocess.run(["docker", "start", "postgresql"]).returncode != 0:
        subprocess.run([
            "docker", "run",
            "--name", "postgresql",
            "-e", "POSTGRES_USER=user",
            "-e", "POSTGRES_PASSWORD=password",
            "-e", "POSTGRES_DB=database",
            "-p", "5432:5432",
            # "-v", f"{os.getcwd()}/postgresql:/var/lib/postgresql/data",
            "-d", "postgres:16"
        ], check=True)

    time.sleep(15)

    subprocess.run([
        "docker", "cp",
        snapshot_path,              
        "postgresql:/tmp/fraudData_snapshot.dump"
    ], check=True)

    time.sleep(15)

    subprocess.run([
        "docker", "exec",
        "-e", "PGPASSWORD=password",
        "postgresql",
        "pg_restore",
        "--clean",
        "--if-exists",
        "-U", "user",
        "-d", "database",
        "/tmp/fraudData_snapshot.dump"
    ], check=True)
    
    print("Database restored from snapshot successfully.")

if __name__ == "__main__":
    main()