from datetime import datetime


def create_report(summary, file_path):
    """
    Save report as a text file.
    """

    with open(file_path, "w", encoding="utf-8") as file:

        file.write("AI Data Science Multi-Agent Report\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Generated : {datetime.now()}\n\n")

        file.write(summary)

    return file_path