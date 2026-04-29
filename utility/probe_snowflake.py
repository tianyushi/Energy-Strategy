"""Probe what's available on the connected Snowflake account."""
from .snowflake_client import SnowflakeClient


def main() -> None:
    with SnowflakeClient() as sf:
        print("=== whoami ===")
        for k, v in sf.whoami().items():
            print(f"  {k:>10}: {v}")

        print("\n=== databases visible to current role ===")
        dbs = sf.read_sql("SHOW DATABASES")
        if "name" in dbs.columns:
            print(dbs[["name", "kind", "owner", "origin"]].to_string(index=False))
        else:
            print(dbs.head(20).to_string(index=False))

        print("\n=== warehouses ===")
        whs = sf.read_sql("SHOW WAREHOUSES")
        if "name" in whs.columns:
            cols = [c for c in ["name", "state", "size", "auto_suspend", "auto_resume"] if c in whs.columns]
            print(whs[cols].to_string(index=False))
        else:
            print(whs.head().to_string(index=False))


if __name__ == "__main__":
    main()
