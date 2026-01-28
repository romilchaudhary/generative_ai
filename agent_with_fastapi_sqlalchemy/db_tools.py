from langchain_community.tools import tool
from database import SessionLocal, User

@tool
def get_user_by_email_tool(email: str) -> str:
    """Tool that retrieves a user by email from the database."""
    db = SessionLocal()
    user = db.query(User).filter(User.email == email).first()
    db.close()
    if user:
        return str(f"User found: ID={user.id}, Name={user.name}, Email={user.email}")
    else:
        return "User not found."