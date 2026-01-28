from langchain_community.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression"""
    return str(eval(expression))

@tool
def company_info(name: str) -> str:
    """Get company info"""
    data = {
        "tata": "Tata Group is an Indian multinational conglomerate.",
        "google": "Google is a global technology company."
    }
    return data.get(name.lower(), "Company not found")