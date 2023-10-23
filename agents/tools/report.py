from langchain.tools import StructuredTool
from pydantic import BaseModel
import os


def write_report(filename, html):
    # Create the 'reports' directory if it doesn't exist
    if not os.path.exists('reports'):
        os.makedirs('reports')

    # Combine the folder name 'reports' with the filename
    full_path = os.path.join('reports', filename)

    with open(full_path, 'w') as f:
        f.write(html)


class WriteReportArgsSchema(BaseModel):
    filename: str
    html: str


write_report_tool = StructuredTool.from_function(
    name="write_report",
    description="Write a HTML file to disk, Use this when someone asks for a report or file",
    func=write_report,
    args_schema=WriteReportArgsSchema
)
