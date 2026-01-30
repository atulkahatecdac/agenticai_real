# First go to https://indianapi.in/documentation/jobs-api
# Register and get an API key
# Add to .env as INDIAN_JOBS_API_KEY
from mcp.server.fastmcp import FastMCP
import requests
import os
from dotenv import load_dotenv

mcp = FastMCP("JobSearch")

@mcp.tool()
def search_jobs(
    limit: str = "10",
    location: str = None,
    title: str = None,
    company: str = None,
    experience: str = None,
    job_type: str = None
) -> dict:
    """
    Search for jobs using the Indian Jobs API.

    Args:
        limit (string): Number of job listings to return.
        location (string, optional): Filter by location.
        title (string, optional): Filter by job title.
        company (string, optional): Filter by company name.
        experience (string, optional): Filter by experience level.
        job_type (string, optional): Filter by job type (Full Time / Part Time).

    Returns:
        dict: API response containing job listings or error info.
    """
    
    load_dotenv()

    try:
        api_key = os.getenv("INDIAN_JOBS_API_KEY")
        if not api_key:
            return {"error": "INDIAN_JOBS_API_KEY not found in environment variables"}

        url = "https://jobs.indianapi.in/jobs"

        # Build query parameters dynamically
        params = {"limit": limit}

        if location:
            params["location"] = location
        if title:
            params["title"] = title
        if company:
            params["company"] = company
        if experience:
            params["experience"] = experience
        if job_type:
            params["job_type"] = job_type

        headers = {
            "X-Api-Key": api_key
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        return response.json()

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run()
