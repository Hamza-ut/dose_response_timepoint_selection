## DEPENDENCIES
try:
    import sys
    from pathlib import Path
    from datetime import datetime
    from time import perf_counter
    from drc_timepoint import load_config, run_analysis_from_config
    from drc_timepoint.logging_utils import setup_logger
except ModuleNotFoundError as e:
    missing_module = e.name
    print(f"❌ Required module not found: '{missing_module}'.")
    sys.exit(1)
except ImportError as e:
    # For C-extension or version issues
    print(f"❌ ImportError: {e}")
    print("   This may happen if the package is partially installed or incompatible.")
    sys.exit(1)


## EXECUTE
def main() -> None:
    # Setup logger
    logger = setup_logger(log_to_file=False)

    # ----------------------------
    # 1. Parse CLI args
    # ----------------------------
    if len(sys.argv) < 2:
        logger.error(
            "HowToRun: python <drc_timepoint_composite_score.py> <path/to/config_file.json>"
        )
        sys.exit(1)

    config_file = Path(sys.argv[1])
    logger.info(f"🔧 Loading config file: {config_file}\n")

    try:
        # ----------------------------
        # 2. Load config
        # ----------------------------
        config = load_config(config_file)
        logger.info("✅ Success: configuration loaded.\n")

        # ----------------------------
        # 3. Run analysis (CORE CALL)
        # ----------------------------
        start = perf_counter()
        logger.info("🚀 Starting composite score analysis...\n")

        result_df = run_analysis_from_config(config)

        logger.info("✅ Analysis completed.\n")

        # ----------------------------
        # 4. Export (optional)
        # ----------------------------
        if config.get("export_results", False):
            input_path = Path(config["file_path"])
            output_file = (
                input_path.parent
                / f"{input_path.stem}_composite_score_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            result_df.to_csv(output_file, index=False)
            logger.info(f"💾 Result exported to CSV: {output_file}\n")
        else:
            logger.info("📌 Export not requested.\n")

        # ----------------------------
        # 5. Print results
        # ----------------------------
        logger.info("🎯 Result:\n%s\n", result_df)

        elapsed = round(perf_counter() - start, 3)
        logger.info(f"🕐 Program finished in {elapsed} seconds.\n")

    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    # input("Press any key to exit...")
